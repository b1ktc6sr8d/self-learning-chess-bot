#!/usr/bin/env python3
"""
♛ Chess RL Web Studio — Graphical Interactive Frontend
Play against your reinforcement-learning trained chess model directly in your browser.

Features:
- Graphical interactive chessboard with click-to-move, drag-and-drop, and legal move indicators.
- Live Value Head evaluation bar (-1.0 to +1.0) and win probability estimation.
- Live Policy Head top-3 candidate move thoughts with probability percentages.
- Checkpoint selector dropdown (switch between model_ep00250, ep00500, ep01000, ep02000, etc.).
- Temperature slider for bot play style (ruthless vs creative).
- Move history table in standard algebraic notation.
- Automatic PGN recording to games/human/human_vs_bot_YYYYMMDD_HHMMSS.pgn for every completed match!
- Zero external pip web dependencies (built on Python's standard http.server).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import chess
import chess.pgn
import torch

from encoding import encode_board, get_legal_move_mask, index_to_move, move_to_index
from model import ChessNet
from search import select_move_2ply

# ── Global State for the active Web Game ──────────────────────────────────
class ChessGameSession:
    def __init__(self, checkpoint_path: str | None = None, filters: int = 128, res_blocks: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filters = filters
        self.res_blocks = res_blocks
        self.model = ChessNet(num_filters=filters, num_res_blocks=res_blocks).to(self.device)
        self.active_checkpoint = None
        self.active_episode = "?"
        self.use_2ply = True
        self.max_candidates = 16

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            # Look for latest checkpoint
            ckpts = sorted(glob.glob("checkpoints/model_ep*.pt"))
            if ckpts:
                self.load_checkpoint(ckpts[-1])

        self.reset_game(human_color="white", temperature=0.1)

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.active_checkpoint = path
            self.active_episode = str(ckpt.get("episode", os.path.basename(path).replace(".pt", "")))
            print(f"Loaded checkpoint {path} (Episode {self.active_episode})")
            return True
        except Exception as e:
            print(f"Error loading checkpoint {path}: {e}")
            return False

    def reset_game(self, human_color: str = "white", temperature: float = 0.1) -> dict:
        self.board = chess.Board()
        self.human_color = chess.WHITE if human_color.lower() == "white" else chess.BLACK
        self.temperature = float(temperature)
        self.move_history_san = []
        self.last_move = None
        self.bot_eval = 0.0
        self.bot_win_pct = 50
        self.bot_top_moves = []
        self.saved_pgn_file = None

        # PGN recording setup
        self.start_time = datetime.datetime.now()
        self.pgn_game = chess.pgn.Game()
        self.pgn_game.headers["Event"] = "Human vs ChessRL Match"
        self.pgn_game.headers["Site"] = "Local Web GUI Studio"
        self.pgn_game.headers["Date"] = self.start_time.strftime("%Y.%m.%d")
        self.pgn_game.headers["Time"] = self.start_time.strftime("%H:%M:%S")
        bot_label = f"ChessRL-Ep{self.active_episode}" if self.active_checkpoint else "ChessRL-Untrained"
        self.pgn_game.headers["White"] = "Human" if self.human_color == chess.WHITE else bot_label
        self.pgn_game.headers["Black"] = bot_label if self.human_color == chess.WHITE else "Human"
        self.pgn_game.headers["GameType"] = "Human Match"
        if self.active_checkpoint:
            self.pgn_game.headers["ModelCheckpoint"] = self.active_checkpoint
        self.pgn_node = self.pgn_game

        # If bot plays white, it makes the first move
        if self.human_color == chess.BLACK:
            self.make_bot_move()

        return self.get_state()

    @torch.no_grad()
    def make_bot_move(self) -> None:
        if self.board.is_game_over(claim_draw=True):
            return

        self.model.eval()

        if self.use_2ply:
            chosen_move, _action_idx, best_score, details = select_move_2ply(
                model=self.model,
                board=self.board,
                device=self.device,
                temperature=self.temperature,
                max_m1=self.max_candidates,
                max_m2=self.max_candidates,
            )
            self.bot_eval = round(best_score, 3)
            self.bot_win_pct = int(max(0, min(100, (self.bot_eval + 1.0) / 2.0 * 100)))

            top_candidates = []
            for d in details[:3]:
                txt = d["san"]
                if d.get("counter_san"):
                    txt += f" → {d['counter_san']}"
                top_candidates.append({
                    "san": txt,
                    "prob": round(d.get("score", 0.0), 3),
                })
            self.bot_top_moves = top_candidates
        else:
            state = encode_board(self.board).unsqueeze(0).to(self.device)
            mask = get_legal_move_mask(self.board).to(self.device)

            logits, value = self.model(state)
            logits = logits.squeeze(0)
            self.bot_eval = round(value.squeeze().item(), 3)
            self.bot_win_pct = int((self.bot_eval + 1.0) / 2.0 * 100)

            # Mask illegal moves
            logits[mask == 0] = float("-inf")

            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=0)
            else:
                probs = torch.zeros_like(logits)
                probs[logits.argmax()] = 1.0

            # Top legal candidate moves
            top_candidates = []
            legal_indices = torch.where(mask > 0)[0]
            sub_probs = probs[legal_indices]
            k = min(3, len(legal_indices))
            top_vals, top_idx_in_legal = torch.topk(sub_probs, k=k)

            for val, idx in zip(top_vals, top_idx_in_legal):
                act_idx = legal_indices[idx].item()
                m = index_to_move(act_idx, self.board)
                if m:
                    top_candidates.append({"san": self.board.san(m), "prob": round(val.item() * 100, 1)})
            self.bot_top_moves = top_candidates

            # Sample action
            if self.temperature > 0:
                action_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                action_idx = logits.argmax().item()

            chosen_move = index_to_move(action_idx, self.board)
            if chosen_move is None or chosen_move not in self.board.legal_moves:
                chosen_move = next(iter(self.board.legal_moves))

        san_move = self.board.san(chosen_move)
        self.last_move = {"uci": chosen_move.uci(), "san": san_move, "player": "bot"}
        self.move_history_san.append(san_move)

        self.pgn_node = self.pgn_node.add_variation(chosen_move)
        self.board.push(chosen_move)

        if self.board.is_game_over(claim_draw=True):
            self.save_completed_pgn()

    def play_human_move(self, uci_str: str) -> dict:
        if self.board.is_game_over(claim_draw=True):
            return {"error": "Game is already over"}

        try:
            move = chess.Move.from_uci(uci_str)
        except Exception:
            return {"error": "Invalid UCI move string"}

        # Handle automatic queen promotion if unspecified
        if move not in self.board.legal_moves:
            promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promo_move in self.board.legal_moves:
                move = promo_move

        if move not in self.board.legal_moves:
            return {"error": f"Illegal move: {uci_str}"}

        san_move = self.board.san(move)
        self.last_move = {"uci": move.uci(), "san": san_move, "player": "human"}
        self.move_history_san.append(san_move)

        self.pgn_node = self.pgn_node.add_variation(move)
        self.board.push(move)

        # Check if game ended on human move
        if self.board.is_game_over(claim_draw=True):
            self.save_completed_pgn()
        else:
            # Bot replies
            self.make_bot_move()

        return self.get_state()

    def save_completed_pgn(self) -> str | None:
        if self.saved_pgn_file:
            return self.saved_pgn_file
        res = self.board.result(claim_draw=True)
        self.pgn_game.headers["Result"] = res
        save_dir = os.path.join("games", "human")
        os.makedirs(save_dir, exist_ok=True)
        stamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"human_vs_bot_{stamp}.pgn"
        full_path = os.path.join(save_dir, filename)
        with open(full_path, "w", encoding="utf-8") as fh:
            print(self.pgn_game, file=fh, end="\n\n")
        self.saved_pgn_file = full_path
        print(f"Recorded Human Match PGN → {full_path}")
        return full_path

    def get_state(self) -> dict:
        is_over = self.board.is_game_over(claim_draw=True)
        result = self.board.result(claim_draw=True) if is_over else None

        status_text = ""
        if is_over:
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                status_text = f"Checkmate! {winner} wins ({result})"
            elif self.board.is_stalemate():
                status_text = f"Draw by Stalemate ({result})"
            elif self.board.is_insufficient_material():
                status_text = f"Draw by Insufficient Material ({result})"
            elif self.board.can_claim_threefold_repetition():
                status_text = f"Draw by Threefold Repetition ({result})"
            elif self.board.can_claim_fifty_moves():
                status_text = f"Draw by 50-Move Rule ({result})"
            else:
                status_text = f"Game Over ({result})"
        elif self.board.is_check():
            side = "White" if self.board.turn == chess.WHITE else "Black"
            status_text = f"Check! ({side} to move)"
        else:
            side = "White" if self.board.turn == chess.WHITE else "Black"
            status_text = f"{side} to move"

        # Legal moves list
        legal_moves = [m.uci() for m in self.board.legal_moves]

        # Available checkpoints
        ckpts = sorted(glob.glob("checkpoints/model_ep*.pt"), reverse=True)

        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_color": "white" if self.human_color == chess.WHITE else "black",
            "is_game_over": is_over,
            "result": result,
            "status_text": status_text,
            "legal_moves": legal_moves,
            "last_move": self.last_move,
            "move_history": self.move_history_san,
            "bot_eval": self.bot_eval,
            "bot_win_pct": self.bot_win_pct,
            "bot_top_moves": self.bot_top_moves,
            "saved_pgn_file": self.saved_pgn_file,
            "active_checkpoint": self.active_checkpoint,
            "active_episode": self.active_episode,
            "available_checkpoints": [os.path.basename(c) for c in ckpts],
        }


# ── Embedded HTML / CSS / JS Frontend ─────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>♛ Chess RL — Interactive Studio</title>
  <style>
    :root {
      --bg-dark: #121417;
      --bg-card: #1c2026;
      --bg-card-hover: #262c35;
      --border-color: #2d3540;
      --accent: #3498db;
      --accent-glow: rgba(52, 152, 219, 0.35);
      --success: #2ecc71;
      --danger: #e74c3c;
      --warning: #f39c12;
      --text: #ecf0f1;
      --text-muted: #8b9bb4;
      --board-light: #dee3e6;
      --board-dark: #8ca2ad;
      --sq-highlight: rgba(241, 196, 15, 0.45);
      --sq-dest: rgba(46, 204, 113, 0.5);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }
    body { background-color: var(--bg-dark); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }

    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand { display: flex; align-items: center; gap: 10px; font-size: 1.25rem; font-weight: 700; color: #fff; }
    .brand span { color: var(--accent); }
    .badge { background: #263342; color: var(--accent); padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }

    .main-container {
      display: flex;
      flex: 1;
      gap: 24px;
      padding: 24px;
      max-width: 1400px;
      margin: 0 auto;
      width: 100%;
    }

    /* Board Area */
    .board-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }
    .board-wrapper {
      position: relative;
      display: inline-block;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
      border-radius: 8px;
      overflow: hidden;
      border: 4px solid #2a313d;
    }
    .chessboard {
      display: grid;
      grid-template-columns: repeat(8, 68px);
      grid-template-rows: repeat(8, 68px);
      width: 544px;
      height: 544px;
      user-select: none;
    }
    .square {
      width: 68px;
      height: 68px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      cursor: pointer;
      font-size: 42px;
      transition: background-color 0.1s ease;
    }
    .square.light { background-color: var(--board-light); }
    .square.dark { background-color: var(--board-dark); }
    .square.selected { background-color: var(--sq-highlight) !important; }
    .square.last-from, .square.last-to { background-color: rgba(243, 156, 18, 0.35) !important; }
    .square.legal-dest::after {
      content: "";
      width: 22px;
      height: 22px;
      background: var(--sq-dest);
      border-radius: 50%;
      position: absolute;
      pointer-events: none;
    }
    .square.legal-dest.has-piece::after {
      width: 62px;
      height: 62px;
      border: 4px solid var(--danger);
      background: transparent;
      border-radius: 50%;
    }

    /* Rank & File labels */
    .coord-rank { position: absolute; left: 3px; top: 3px; font-size: 11px; font-weight: 700; opacity: 0.6; pointer-events: none; }
    .coord-file { position: absolute; right: 3px; bottom: 3px; font-size: 11px; font-weight: 700; opacity: 0.6; pointer-events: none; }
    .light .coord-rank, .light .coord-file { color: var(--board-dark); }
    .dark .coord-rank, .dark .coord-file { color: var(--board-light); }

    .piece-img {
      width: 58px;
      height: 58px;
      pointer-events: none;
      transition: transform 0.1s ease;
    }

    /* Status Bar */
    .status-bar {
      width: 544px;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 1rem;
      font-weight: 600;
    }
    .status-text { display: flex; align-items: center; gap: 8px; }
    .status-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--success); }
    .status-dot.check { background: var(--danger); animation: pulse 1s infinite; }

    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
      70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(231, 76, 60, 0); }
      100% { transform: scale(0.95); }
    }

    /* Sidebar Panels */
    .sidebar {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-width: 380px;
    }
    .panel {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px;
    }
    .panel-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Eval Bar & AI thoughts */
    .eval-container {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .eval-meter {
      height: 28px;
      background: #2a313d;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
      display: flex;
    }
    .eval-fill {
      height: 100%;
      background: linear-gradient(90deg, #e67e22, #f1c40f);
      transition: width 0.4s ease;
    }
    .eval-label {
      position: absolute;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      font-size: 0.85rem;
      font-weight: 700;
      color: #fff;
      text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }
    .thought-chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .chip {
      background: #262e38;
      border: 1px solid #333f4d;
      padding: 6px 12px;
      border-radius: 16px;
      font-size: 0.82rem;
      font-weight: 600;
      color: #cbd5e1;
    }
    .chip span { color: var(--accent); margin-left: 4px; }

    /* Move History Table */
    .history-table {
      height: 170px;
      overflow-y: auto;
      background: #14171c;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      font-size: 0.9rem;
    }
    .history-row {
      display: grid;
      grid-template-columns: 48px 1fr 1fr;
      padding: 6px 10px;
      border-bottom: 1px solid #1e242d;
    }
    .history-row:nth-child(even) { background: rgba(255,255,255,0.02); }
    .history-num { color: var(--text-muted); font-weight: 600; }
    .history-move { font-family: monospace; font-size: 0.95rem; }

    /* Controls */
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      background: #2c3e50;
      border: 1px solid #34495e;
      color: #fff;
      padding: 10px 14px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    button:hover { background: #34495e; border-color: var(--accent); }
    button.primary { background: var(--accent); border-color: #2980b9; }
    button.primary:hover { background: #2980b9; }

    select, input[type=range] {
      width: 100%;
      background: #14171c;
      border: 1px solid var(--border-color);
      color: #fff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    .form-group {
      margin-bottom: 12px;
    }
    .form-label {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }

    /* Modal for Promotion */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0,0,0,0.7);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }
    .modal-overlay.active { display: flex; }
    .modal-card {
      background: var(--bg-card);
      border: 2px solid var(--accent);
      padding: 24px;
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      align-items: center;
    }
    .promo-pieces { display: flex; gap: 14px; font-size: 48px; }
    .promo-opt {
      width: 64px; height: 64px;
      background: #262e38;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.1s;
    }
    .promo-opt:hover { background: var(--accent); transform: scale(1.08); }

    .saved-alert {
      background: rgba(46, 204, 113, 0.15);
      border: 1px solid var(--success);
      color: #a8f0c6;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      display: none;
      margin-top: 10px;
      line-height: 1.4;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      ♛ <span>Chess RL</span> Studio
      <div class="badge" id="botBadge">Ep 2000</div>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);">Pure Self-Play Reinforcement Learning</div>
  </header>

  <div class="main-container">
    <!-- Board -->
    <div class="board-container">
      <div class="board-wrapper">
        <div class="chessboard" id="chessboard"></div>
      </div>
      <div class="status-bar">
        <div class="status-text">
          <div class="status-dot" id="statusDot"></div>
          <span id="statusMessage">White to move</span>
        </div>
        <div id="turnIndicator" style="font-size: 0.85rem; color: var(--text-muted);">Turn: White</div>
      </div>
    </div>

    <!-- Sidebar -->
    <div class="sidebar">
      <!-- Live AI Telemetry -->
      <div class="panel">
        <div class="panel-title">
          <span>Neural AI Telemetry</span>
          <span id="evalScalar" style="color: var(--accent); font-weight: 700;">+0.00</span>
        </div>
        <div class="eval-container">
          <div class="eval-meter">
            <div class="eval-fill" id="evalFill" style="width: 50%;"></div>
            <div class="eval-label">
              <span id="whiteOdds">White 50%</span>
              <span id="blackOdds">Black 50%</span>
            </div>
          </div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">Top Move Probabilities (Policy Head):</div>
          <div class="thought-chips" id="thoughtChips">
            <div class="chip">Analyzing position...</div>
          </div>
        </div>
      </div>

      <!-- Move History -->
      <div class="panel">
        <div class="panel-title">Move History</div>
        <div class="history-table" id="historyTable"></div>
      </div>

      <!-- Controls & Settings -->
      <div class="panel">
        <div class="panel-title">Match Controls</div>

        <div class="form-group">
          <div class="form-label">
            <span>Model Checkpoint</span>
            <span style="color: var(--accent);">Active</span>
          </div>
          <select id="checkpointSelect"></select>
        </div>

        <div class="form-group">
          <div class="form-label">
            <span>Bot Temperature (Greedy vs Creative)</span>
            <span id="tempVal">0.10</span>
          </div>
          <input type="range" id="tempSlider" min="0.0" max="1.5" step="0.05" value="0.10">
        </div>

        <div class="controls-grid" style="margin-bottom: 10px;">
          <button class="primary" id="btnNewWhite">Play as White</button>
          <button class="primary" id="btnNewBlack">Play as Black</button>
        </div>

        <div class="controls-grid">
          <button id="btnFlip">Flip Board</button>
          <button id="btnDownloadPgn">Download PGN</button>
        </div>

        <div class="saved-alert" id="savedAlert"></div>
      </div>
    </div>
  </div>

  <!-- Promotion Dialog -->
  <div class="modal-overlay" id="promoModal">
    <div class="modal-card">
      <div style="font-size: 1.1rem; font-weight: 700;">Promote Pawn</div>
      <div class="promo-pieces">
        <div class="promo-opt" data-piece="q">♕</div>
        <div class="promo-opt" data-piece="r">♖</div>
        <div class="promo-opt" data-piece="b">♗</div>
        <div class="promo-opt" data-piece="n">♘</div>
      </div>
    </div>
  </div>

  <script>
    // SVG piece dictionary
    const PIECE_SVGS = {
      'P': '<svg viewBox="0 0 45 45"><path d="m22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></svg>',
      'N': '<svg viewBox="0 0 45 45"><path d="m22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#fff" stroke="#000" stroke-width="1.5"/><path d="m24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.04-.94-1.48-4.19-.5-5.5 2-2.5 5.5-3.5 7-6 1.5-2.5.5-4.5.5-4.5z" fill="#fff" stroke="#000" stroke-width="1.5"/></svg>',
      'B': '<svg viewBox="0 0 45 45"><g fill="none" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.46 3-2 3-2z" fill="#fff"/><path d="m15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z" fill="#fff"/><path d="m25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z" fill="#fff"/></g></svg>',
      'R': '<svg viewBox="0 0 45 45"><g fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m9 39h27v-3H9v3zm3-3v-4.5h21V36H12zm2-4.5 1.5-16.5h14l1.5 16.5H14zM11 14h23l-2-6H13l-2 6zm3-6V4h4v4h3V4h4v4h3V4h4v4H14z"/></g></svg>',
      'Q': '<svg viewBox="0 0 45 45"><g fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m9 26c8.5-1.5 21-1.5 27 0l2-12-7 11-4-14-4 14-7-11 2 12zm0 0c0 4 1 7 4 8h19c3-1 4-4 4-8H9zm-2 13h31v-3H7v3z"/><circle cx="6" cy="12" r="2"/><circle cx="14" cy="9" r="2"/><circle cx="22.5" cy="8" r="2"/><circle cx="31" cy="9" r="2"/><circle cx="39" cy="12" r="2"/></g></svg>',
      'K': '<svg viewBox="0 0 45 45"><g fill="none" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m22.5 11.63V6M20 8h5" stroke-linejoin="miter"/><path d="m22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#fff"/><path d="m11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-1-1 8-6 8-2 0-2-3-4-3-2 0-2 3-4 3s-2-3-4-3c-2 0-2 3-4 3-5 0-2-9-6-8-3 6 6 10.5 6 10.5v7z" fill="#fff"/><path d="m11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"/></g></svg>',
      'p': '<svg viewBox="0 0 45 45"><path d="m22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#2c3e50" stroke="#000" stroke-width="1.5"/></svg>',
      'n': '<svg viewBox="0 0 45 45"><path d="m22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#2c3e50" stroke="#000" stroke-width="1.5"/><path d="m24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.04-.94-1.48-4.19-.5-5.5 2-2.5 5.5-3.5 7-6 1.5-2.5.5-4.5.5-4.5z" fill="#2c3e50" stroke="#000" stroke-width="1.5"/></svg>',
      'b': '<svg viewBox="0 0 45 45"><g fill="none" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.46 3-2 3-2z" fill="#2c3e50"/><path d="m15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z" fill="#2c3e50"/><path d="m25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z" fill="#2c3e50"/></g></svg>',
      'r': '<svg viewBox="0 0 45 45"><g fill="#2c3e50" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m9 39h27v-3H9v3zm3-3v-4.5h21V36H12zm2-4.5 1.5-16.5h14l1.5 16.5H14zM11 14h23l-2-6H13l-2 6zm3-6V4h4v4h3V4h4v4h3V4h4v4H14z"/></g></svg>',
      'q': '<svg viewBox="0 0 45 45"><g fill="#2c3e50" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m9 26c8.5-1.5 21-1.5 27 0l2-12-7 11-4-14-4 14-7-11 2 12zm0 0c0 4 1 7 4 8h19c3-1 4-4 4-8H9zm-2 13h31v-3H7v3z"/><circle cx="6" cy="12" r="2"/><circle cx="14" cy="9" r="2"/><circle cx="22.5" cy="8" r="2"/><circle cx="31" cy="9" r="2"/><circle cx="39" cy="12" r="2"/></g></svg>',
      'k': '<svg viewBox="0 0 45 45"><g fill="none" stroke="#000" stroke-width="1.5" stroke-linecap="round"><path d="m22.5 11.63V6M20 8h5" stroke-linejoin="miter"/><path d="m22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#2c3e50"/><path d="m11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-1-1 8-6 8-2 0-2-3-4-3-2 0-2 3-4 3s-2-3-4-3c-2 0-2 3-4 3-5 0-2-9-6-8-3 6 6 10.5 6 10.5v7z" fill="#2c3e50"/><path d="m11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"/></g></svg>'
    };

    let gameState = null;
    let selectedSq = null;
    let boardFlipped = false;
    let pendingPromo = null;

    const boardEl = document.getElementById("chessboard");

    async function fetchState() {
      const res = await fetch("/api/state");
      const data = await res.json();
      updateUI(data);
    }

    async function sendMove(from, to, promo = "") {
      const uci = from + to + promo;
      const res = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ move: uci })
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
      } else {
        updateUI(data);
      }
    }

    function parseFEN(fen) {
      const rows = fen.split(" ")[0].split("/");
      const grid = [];
      for (let r = 0; r < 8; r++) {
        const row = [];
        for (let ch of rows[r]) {
          if (!isNaN(ch)) {
            for (let i = 0; i < parseInt(ch); i++) row.push(null);
          } else {
            row.push(ch);
          }
        }
        grid.push(row);
      }
      return grid;
    }

    function renderBoard() {
      if (!gameState) return;
      boardEl.innerHTML = "";
      const grid = parseFEN(gameState.fen);

      const ranks = boardFlipped ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
      const files = boardFlipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];

      for (let rIdx = 0; rIdx < 8; rIdx++) {
        const r = ranks[rIdx];
        for (let fIdx = 0; fIdx < 8; fIdx++) {
          const f = files[fIdx];
          const sqName = String.fromCharCode(97 + f) + (r + 1);
          const isLight = (r + f) % 2 !== 0;

          const sqEl = document.createElement("div");
          sqEl.className = `square ${isLight ? "light" : "dark"}`;
          sqEl.dataset.square = sqName;

          if (selectedSq === sqName) sqEl.classList.add("selected");
          if (gameState.last_move) {
            if (gameState.last_move.uci.startsWith(sqName)) sqEl.classList.add("last-from");
            if (gameState.last_move.uci.substring(2, 4) === sqName) sqEl.classList.add("last-to");
          }

          // Check if legal destination for selected square
          if (selectedSq) {
            const hasLegalMove = gameState.legal_moves.some(m => m.startsWith(selectedSq) && m.substring(2, 4) === sqName);
            if (hasLegalMove) {
              sqEl.classList.add("legal-dest");
              if (grid[7 - r][f]) sqEl.classList.add("has-piece");
            }
          }

          // Coordinates
          if (fIdx === 0) {
            const rankLabel = document.createElement("div");
            rankLabel.className = "coord-rank";
            rankLabel.innerText = (r + 1);
            sqEl.appendChild(rankLabel);
          }
          if (rIdx === 7) {
            const fileLabel = document.createElement("div");
            fileLabel.className = "coord-file";
            fileLabel.innerText = String.fromCharCode(97 + f);
            sqEl.appendChild(fileLabel);
          }

          // Piece
          const p = grid[7 - r][f];
          if (p && PIECE_SVGS[p]) {
            const pDiv = document.createElement("div");
            pDiv.className = "piece-img";
            pDiv.innerHTML = PIECE_SVGS[p];
            sqEl.appendChild(pDiv);
          }

          sqEl.onclick = () => onSquareClick(sqName, p);
          boardEl.appendChild(sqEl);
        }
      }
    }

    function onSquareClick(sq, piece) {
      if (gameState.is_game_over) return;

      const isHumanTurn = (gameState.turn === gameState.human_color);
      if (!isHumanTurn) return;

      if (!selectedSq) {
        // Must select own piece
        if (!piece) return;
        const isWhitePiece = piece === piece.toUpperCase();
        const isMyPiece = (gameState.human_color === "white" && isWhitePiece) || (gameState.human_color === "black" && !isWhitePiece);
        if (isMyPiece) {
          selectedSq = sq;
          renderBoard();
        }
      } else {
        if (selectedSq === sq) {
          selectedSq = null;
          renderBoard();
          return;
        }

        // Check if legal move
        const matches = gameState.legal_moves.filter(m => m.startsWith(selectedSq) && m.substring(2, 4) === sq);
        if (matches.length > 0) {
          // Check for promotion
          const isPromo = matches.some(m => m.length === 5);
          if (isPromo) {
            pendingPromo = { from: selectedSq, to: sq };
            document.getElementById("promoModal").classList.add("active");
            return;
          } else {
            const from = selectedSq;
            selectedSq = null;
            sendMove(from, sq);
          }
        } else {
          // Clicked another own piece?
          if (piece) {
            const isWhitePiece = piece === piece.toUpperCase();
            const isMyPiece = (gameState.human_color === "white" && isWhitePiece) || (gameState.human_color === "black" && !isWhitePiece);
            if (isMyPiece) {
              selectedSq = sq;
              renderBoard();
              return;
            }
          }
          selectedSq = null;
          renderBoard();
        }
      }
    }

    function updateUI(data) {
      gameState = data;
      boardFlipped = (data.human_color === "black");

      renderBoard();

      // Status
      const statusMsg = document.getElementById("statusMessage");
      const statusDot = document.getElementById("statusDot");
      const turnIndicator = document.getElementById("turnIndicator");
      statusMsg.innerText = data.status_text;
      turnIndicator.innerText = "Turn: " + (data.turn.charAt(0).toUpperCase() + data.turn.slice(1));
      if (data.status_text.includes("Check")) {
        statusDot.className = "status-dot check";
      } else {
        statusDot.className = "status-dot";
      }

      // Eval Bar
      const evalFill = document.getElementById("evalFill");
      const evalScalar = document.getElementById("evalScalar");
      const whiteOdds = document.getElementById("whiteOdds");
      const blackOdds = document.getElementById("blackOdds");

      let wPct = data.bot_win_pct;
      if (data.human_color === "black") {
        // bot is white
        wPct = data.bot_win_pct;
      } else {
        // bot is black
        wPct = 100 - data.bot_win_pct;
      }
      const bPct = 100 - wPct;

      evalFill.style.width = wPct + "%";
      evalScalar.innerText = (data.bot_eval >= 0 ? "+" : "") + data.bot_eval.toFixed(2);
      whiteOdds.innerText = "White " + wPct + "%";
      blackOdds.innerText = "Black " + bPct + "%";

      // Bot Thought Chips
      const chipsEl = document.getElementById("thoughtChips");
      chipsEl.innerHTML = "";
      if (data.bot_top_moves && data.bot_top_moves.length > 0) {
        data.bot_top_moves.forEach(m => {
          const chip = document.createElement("div");
          chip.className = "chip";
          const scoreText = (Math.abs(m.prob) <= 1.0)
            ? `Q: ${(m.prob >= 0 ? '+' : '') + m.prob.toFixed(2)}`
            : `${m.prob}%`;
          chip.innerHTML = `${m.san} <span>(${scoreText})</span>`;
          chipsEl.appendChild(chip);
        });
      } else {
        chipsEl.innerHTML = `<div class="chip">Ready for next move</div>`;
      }

      // Move History
      const histTable = document.getElementById("historyTable");
      histTable.innerHTML = "";
      for (let i = 0; i < data.move_history.length; i += 2) {
        const moveNum = (i / 2) + 1;
        const wMove = data.move_history[i] || "";
        const bMove = data.move_history[i + 1] || "";
        const row = document.createElement("div");
        row.className = "history-row";
        row.innerHTML = `<span class="history-num">${moveNum}.</span><span class="history-move">${wMove}</span><span class="history-move">${bMove}</span>`;
        histTable.appendChild(row);
      }
      histTable.scrollTop = histTable.scrollHeight;

      // Checkpoint badge & dropdown
      document.getElementById("botBadge").innerText = "Ep " + data.active_episode;
      const ckptSel = document.getElementById("checkpointSelect");
      if (ckptSel.children.length === 0 && data.available_checkpoints) {
        data.available_checkpoints.forEach(c => {
          const opt = document.createElement("option");
          opt.value = c;
          opt.innerText = c;
          if (data.active_checkpoint && data.active_checkpoint.endsWith(c)) opt.selected = true;
          ckptSel.appendChild(opt);
        });
      }

      // Saved Alert
      const alertEl = document.getElementById("savedAlert");
      if (data.saved_pgn_file) {
        alertEl.style.display = "block";
        alertEl.innerHTML = `✓ Match PGN automatically archived to: <br><code>${data.saved_pgn_file}</code>`;
      } else {
        alertEl.style.display = "none";
      }
    }

    // Event listeners
    document.getElementById("btnNewWhite").onclick = async () => {
      selectedSq = null;
      const res = await fetch("/api/new_game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ color: "white", temperature: parseFloat(document.getElementById("tempSlider").value) })
      });
      updateUI(await res.json());
    };

    document.getElementById("btnNewBlack").onclick = async () => {
      selectedSq = null;
      const res = await fetch("/api/new_game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ color: "black", temperature: parseFloat(document.getElementById("tempSlider").value) })
      });
      updateUI(await res.json());
    };

    document.getElementById("btnFlip").onclick = () => {
      boardFlipped = !boardFlipped;
      renderBoard();
    };

    document.getElementById("tempSlider").oninput = (e) => {
      document.getElementById("tempVal").innerText = parseFloat(e.target.value).toFixed(2);
    };

    document.getElementById("checkpointSelect").onchange = async (e) => {
      const val = e.target.value;
      const res = await fetch("/api/select_checkpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint: "checkpoints/" + val })
      });
      const data = await res.json();
      if (data.success) {
        fetchState();
      }
    };

    document.getElementById("btnDownloadPgn").onclick = () => {
      window.location.href = "/api/export_pgn";
    };

    // Promotion selection
    document.querySelectorAll(".promo-opt").forEach(opt => {
      opt.onclick = () => {
        const piece = opt.dataset.piece;
        document.getElementById("promoModal").classList.remove("active");
        if (pendingPromo) {
          const { from, to } = pendingPromo;
          pendingPromo = null;
          selectedSq = null;
          sendMove(from, to, piece);
        }
      };
    });

    // Start
    fetchState();
  </script>
</body>
</html>
"""

# ── HTTP Request Handler ──────────────────────────────────────────────────
class ChessRequestHandler(BaseHTTPRequestHandler):
    session: ChessGameSession = None  # set in main

    def log_message(self, format, *args):
        # Silent clean logging
        pass

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            self.send_json(self.session.get_state())
        elif parsed.path == "/api/export_pgn":
            pgn_str = str(self.session.pgn_game)
            body = pgn_str.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-chess-pgn")
            self.send_header("Content-Disposition", 'attachment; filename="match.pgn"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception:
            payload = {}

        if parsed.path == "/api/move":
            uci_move = payload.get("move", "")
            res = self.session.play_human_move(uci_move)
            self.send_json(res)
        elif parsed.path == "/api/new_game":
            color = payload.get("color", "white")
            temp = payload.get("temperature", 0.1)
            res = self.session.reset_game(human_color=color, temperature=temp)
            self.send_json(res)
        elif parsed.path == "/api/select_checkpoint":
            ckpt = payload.get("checkpoint", "")
            success = self.session.load_checkpoint(ckpt)
            self.send_json({"success": success})
        else:
            self.send_response(404)
            self.end_headers()


def run_web_gui(port: int = 5000, checkpoint: str | None = None) -> None:
    session = ChessGameSession(checkpoint_path=checkpoint)
    ChessRequestHandler.session = session

    server = HTTPServer(("127.0.0.1", port), ChessRequestHandler)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("      ♛  CHESS RL INTERACTIVE WEB STUDIO  ♛")
    print("=" * 65)
    print(f"   Server active at:     {url}")
    print(f"   Active Checkpoint:    {session.active_checkpoint or 'Untrained'}")
    print(f"   Human PGN Archive:    games/human/human_vs_bot_*.pgn")
    print("   Press Ctrl+C in this console to stop the server.")
    print("=" * 65)

    # Open browser automatically
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Chess RL Web Server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Chess RL Web GUI Studio")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default 5000)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial model checkpoint .pt")
    args = parser.parse_args()
    run_web_gui(port=args.port, checkpoint=args.checkpoint)
