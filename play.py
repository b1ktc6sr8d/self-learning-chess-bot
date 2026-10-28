#!/usr/bin/env python3
"""
Interactive play script: play against the trained Chess RL model in your terminal.

Usage:
    python play.py                                      # Play as White vs model
    python play.py --color black                        # Play as Black vs model
    python play.py --checkpoint checkpoints/model_ep00100.pt
    python play.py --mode bot-vs-random                 # Benchmark bot vs random moves
"""

from __future__ import annotations

import argparse
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import chess
import chess.pgn
import torch

from encoding import (
    encode_board,
    get_legal_move_mask,
    index_to_move,
    move_to_index,
)
from model import ChessNet
from search import select_move_2ply


UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    ".": "·",
}


def render_board(board: chess.Board, flip: bool = False) -> str:
    """Pretty-print board with Unicode pieces, ranks, and files."""
    lines = []
    lines.append("    a   b   c   d   e   f   g   h")
    lines.append("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")

    ranks = range(8) if flip else range(7, -1, -1)
    files = range(7, -1, -1) if flip else range(8)

    for r in ranks:
        row_str = f"{r + 1} │"
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            sym = UNICODE_PIECES[piece.symbol()] if piece else "·"
            row_str += f" {sym} │"
        row_str += f" {r + 1}"
        lines.append(row_str)
        if (r > 0 and not flip) or (r < 7 and flip):
            lines.append("  ├───┼───┼───┼───┼───┼───┼───┼───┤")

    lines.append("  └───┴───┴───┴───┴───┴───┴───┴───┘")
    lines.append("    a   b   c   d   e   f   g   h")
    return "\n".join(lines)


@torch.no_grad()
def get_bot_move(
    model: ChessNet,
    board: chess.Board,
    device: torch.device,
    temperature: float = 0.1,
    use_2ply: bool = True,
    max_candidates: int = 16,
) -> tuple[chess.Move, float, list]:
    """
    Query the model for the best legal move, win probability, and top suggestions.
    Supports either 2-ply Minimax lookahead or 0-ply direct policy.
    """
    model.eval()

    if use_2ply:
        chosen_move, _action_idx, best_score, details = select_move_2ply(
            model=model,
            board=board,
            device=device,
            temperature=temperature,
            max_m1=max_candidates,
            max_m2=max_candidates,
        )
        return chosen_move, best_score, details

    # 0-ply direct policy fallback
    state = encode_board(board).unsqueeze(0).to(device)
    mask = get_legal_move_mask(board).to(device)

    logits, value = model(state)
    logits = logits.squeeze(0)
    value_scalar = value.squeeze().item()

    # Mask illegal moves
    logits[mask == 0] = float("-inf")

    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=0)
    else:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0

    # Top legal move candidates
    top_moves: list[tuple[str, float]] = []
    legal_indices = torch.where(mask > 0)[0]
    sub_probs = probs[legal_indices]
    k = min(3, len(legal_indices))
    top_vals, top_idx_in_legal = torch.topk(sub_probs, k=k)

    for val, idx in zip(top_vals, top_idx_in_legal):
        act_idx = legal_indices[idx].item()
        m = index_to_move(act_idx, board)
        if m:
            top_moves.append((board.san(m), val.item() * 100))

    # Sample chosen move
    if temperature > 0:
        action_idx = torch.multinomial(probs, num_samples=1).item()
    else:
        action_idx = logits.argmax().item()

    chosen_move = index_to_move(action_idx, board)
    if chosen_move is None or chosen_move not in board.legal_moves:
        chosen_move = next(iter(board.legal_moves))

    return chosen_move, value_scalar, top_moves


def play_game_interactive(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_filters=args.filters, num_res_blocks=args.res_blocks).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ep = ckpt.get("episode", "?")
        print(f"Successfully loaded checkpoint (trained for {ep} episodes).\n")
    else:
        if args.checkpoint:
            print(f"Warning: Checkpoint '{args.checkpoint}' not found.")
        print("Using untrained / initialized model.\n")

    board = chess.Board()
    human_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK

    # ── Setup PGN recording ───────────────────────────────────────────
    import datetime
    start_time = datetime.datetime.now()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Human vs ChessRL Match" if args.mode == "human-vs-bot" else f"ChessRL Match ({args.mode})"
    pgn_game.headers["Site"] = "Local Terminal"
    pgn_game.headers["Date"] = start_time.strftime("%Y.%m.%d")
    pgn_game.headers["Time"] = start_time.strftime("%H:%M:%S")
    bot_label = f"ChessRL (ep {ep})" if args.checkpoint else "ChessRL (untrained)"
    if args.mode == "human-vs-bot":
        pgn_game.headers["White"] = "Human" if human_color == chess.WHITE else bot_label
        pgn_game.headers["Black"] = bot_label if human_color == chess.WHITE else "Human"
        pgn_game.headers["GameType"] = "Human"
    else:
        pgn_game.headers["White"] = bot_label
        pgn_game.headers["Black"] = "Random" if args.mode == "bot-vs-random" else bot_label
        pgn_game.headers["GameType"] = "Benchmark"
    if args.checkpoint:
        pgn_game.headers["ModelCheckpoint"] = str(args.checkpoint)
    pgn_node = pgn_game

    print("=" * 60)
    print("      ♛  CHESS RL INTERACTIVE MATCH  ♛")
    print("=" * 60)
    if args.mode == "human-vs-bot":
        print(f"You are playing as: {args.color.upper()}")
        print("Enter moves in SAN (e.g. e4, Nf3, O-O) or UCI (e.g. e2e4).")
        print("Type 'quit' to exit or 'eval' for board evaluation.")
    else:
        print(f"Mode: {args.mode}")
    print("=" * 60)
    print()

    move_num = 1

    while not board.is_game_over(claim_draw=True):
        print(render_board(board, flip=(human_color == chess.BLACK)))
        print()

        is_human_turn = (args.mode == "human-vs-bot") and (board.turn == human_color)
        side_name = "White" if board.turn == chess.WHITE else "Black"

        if is_human_turn:
            while True:
                try:
                    user_input = input(f"[{side_name} to move] Enter move: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return

                if user_input.lower() in ("quit", "exit"):
                    print("Exiting game.")
                    return

                # Parse move
                move = None
                try:
                    move = board.parse_san(user_input)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(user_input)
                    except ValueError:
                        pass

                if move and move in board.legal_moves:
                    pgn_node = pgn_node.add_variation(move)
                    board.push(move)
                    break
                else:
                    legal_sans = [board.san(m) for m in board.legal_moves]
                    print(f"Illegal move '{user_input}'. Legal moves: {', '.join(legal_sans[:8])}...")
        else:
            # Bot or Random turn
            if args.mode == "bot-vs-random" and board.turn != human_color:
                import random
                bot_move = random.choice(list(board.legal_moves))
                print(f"Random player played: {board.san(bot_move)}")
                pgn_node = pgn_node.add_variation(bot_move)
                board.push(bot_move)
            else:
                use_2ply = getattr(args, "search", "2ply") == "2ply"
                bot_move, value_est, top_moves = get_bot_move(
                    model, board, device,
                    temperature=args.temperature,
                    use_2ply=use_2ply,
                    max_candidates=getattr(args, "max_candidates", 16),
                )
                eval_str = f"{value_est:+.2f}"
                pct = int(max(0, min(100, (value_est + 1.0) / 2.0 * 100)))

                if top_moves and isinstance(top_moves[0], dict):
                    suggestions = ", ".join([
                        f"{d['san']}" + (f" → {d['counter_san']}" if d.get("counter_san") else "") + f" [Q: {d.get('score', 0):+.2f}]"
                        for d in top_moves[:3]
                    ])
                else:
                    suggestions = ", ".join([f"{m} ({p:.0f}%)" for m, p in top_moves])

                search_tag = "2-Ply Minimax" if use_2ply else "0-Ply Policy"
                print(f"Bot ({side_name} via {search_tag}) evaluated position: {eval_str} ({pct}% win estimate)")
                print(f"Top choices considered: {suggestions}")
                print(f"Bot played: {board.san(bot_move)}")
                pgn_node = pgn_node.add_variation(bot_move)
                board.push(bot_move)

        print("-" * 60)
        move_num += 1

    # Game Over
    print()
    print("=" * 60)
    print("                  GAME OVER")
    print("=" * 60)
    print(render_board(board))
    outcome = board.outcome(claim_draw=True)
    res = board.result(claim_draw=True)
    if outcome:
        print(f"Result: {res} ({outcome.termination.name})")
    else:
        print(f"Result: {res}")
    print("=" * 60)

    # ── Save PGN with Date/Time & GameType ────────────────────────────
    pgn_game.headers["Result"] = res
    subfolder = "human" if args.mode == "human-vs-bot" else "benchmark"
    save_dir = os.path.join("games", subfolder)
    os.makedirs(save_dir, exist_ok=True)
    stamp = start_time.strftime("%Y%m%d_%H%M%S")
    game_file = os.path.join(save_dir, f"match_{stamp}.pgn")
    with open(game_file, "w", encoding="utf-8") as fh:
        print(pgn_game, file=fh, end="\n\n")
    print(f"📁 Game successfully recorded → {game_file}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Play against the Chess RL bot")
    ap.add_argument("--color", type=str, default="white", choices=["white", "black"],
                    help="Human player color")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to .pt checkpoint")
    ap.add_argument("--mode", type=str, default="human-vs-bot",
                    choices=["human-vs-bot", "bot-vs-random", "bot-vs-bot"],
                    help="Game mode")
    ap.add_argument("--search", type=str, default="2ply", choices=["2ply", "0ply"],
                    help="Search engine: 2-ply Minimax lookahead or 0-ply policy")
    ap.add_argument("--max-candidates", type=int, default=16,
                    help="Maximum candidate moves per ply evaluated in 2-ply search")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Softmax temperature for bot move selection (0.0 = greedy)")
    ap.add_argument("--filters", type=int, default=128,
                    help="Convolutional filters in model")
    ap.add_argument("--res-blocks", type=int, default=4,
                    help="Residual blocks in model")
    args = ap.parse_args()
    play_game_interactive(args)


if __name__ == "__main__":
    main()
