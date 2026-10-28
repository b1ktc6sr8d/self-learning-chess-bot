"""
Self-play engine — the model plays chess against itself.

Each call to ``play_game`` produces:
1. A list of ``(state_tensor, action_index, value_target)`` training tuples.
2. A PGN record appended to ``<pgn_dir>/training_games.pgn``.

The sole reward signal is sparse:
    +1  checkmate win
    -1  checkmate loss
     0  draw (stalemate, repetition, 50-move, insufficient material,
             or move-limit reached)

python-chess is the *only* authority on move legality.  The network
never decides whether a move is legal — it merely ranks the legal
options provided by the referee.
"""

from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING

import chess
import chess.pgn
import numpy as np
import torch

from encoding import (
    ACTION_SIZE,
    encode_board,
    get_legal_move_mask,
    index_to_move,
    move_to_index,
)
from search import select_move_2ply

if TYPE_CHECKING:
    pass

# Maximum half-moves before the game is declared a draw.
MAX_GAME_MOVES = 512


# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def play_game(
    model: torch.nn.Module,
    device: torch.device,
    temperature: float = 1.0,
    game_id: int = 0,
    pgn_dir: str = "games",
    gamma: float = 0.99,
    use_2ply: bool = True,
    max_candidates: int = 16,
) -> tuple[list[tuple[torch.Tensor, int, float]], str, int]:
    """
    Play one complete self-play game using 2-ply Minimax Lookahead or 0-ply policy.

    Parameters
    ----------
    model : ChessNet
        The current neural network (used for both sides).
    device : torch.device
        CPU or CUDA device for inference.
    temperature : float
        Softmax temperature.  Higher → more exploration.
    game_id : int
        Sequential game number (used in PGN headers).
    pgn_dir : str
        Folder where the PGN file is written.
    gamma : float
        Reward discount factor (0 < gamma <= 1.0). Encourages faster wins.
    use_2ply : bool
        Whether to use 2-ply Minimax lookahead with batched neural evaluation.
    max_candidates : int
        Max candidate moves evaluated per ply in 2-ply search.

    Returns
    -------
    training_data : list[(Tensor, int, float)]
        (encoded_state, action_index, value_target) for every position.
    result : str
        ``"1-0"``, ``"0-1"``, or ``"1/2-1/2"``.
    move_count : int
        Number of half-moves played.
    """
    model.eval()
    board = chess.Board()

    # History: we fill in value_target after the game ends.
    history: list[dict] = []

    # ── PGN scaffold ─────────────────────────────────────────────────
    now = datetime.datetime.now()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"]    = "RL Self-Play Training"
    pgn_game.headers["Site"]     = "Local"
    pgn_game.headers["Date"]     = now.strftime("%Y.%m.%d")
    pgn_game.headers["Time"]     = now.strftime("%H:%M:%S")
    pgn_game.headers["Round"]    = str(game_id)
    pgn_game.headers["White"]    = "ChessRL"
    pgn_game.headers["Black"]    = "ChessRL"
    pgn_game.headers["GameType"] = "Training"
    pgn_node = pgn_game

    move_count = 0

    # ── Game loop ────────────────────────────────────────────────────
    while not board.is_game_over(claim_draw=True) and move_count < MAX_GAME_MOVES:

        if use_2ply:
            chosen_move, action_idx, _q_val, _details = select_move_2ply(
                model=model,
                board=board,
                device=device,
                temperature=temperature,
                max_m1=max_candidates,
                max_m2=max_candidates,
            )
        else:
            state_tensor = encode_board(board).unsqueeze(0).to(device)
            legal_mask   = get_legal_move_mask(board).to(device)

            # Forward pass
            policy_logits, _value = model(state_tensor)
            policy_logits = policy_logits.squeeze(0)          # (4096,)

            # Mask illegal moves → −∞
            policy_logits[legal_mask == 0] = float("-inf")

            # Temperature-scaled softmax
            if temperature > 0:
                probs = torch.softmax(policy_logits / temperature, dim=0)
            else:  # greedy
                probs = torch.zeros_like(policy_logits)
                probs[policy_logits.argmax()] = 1.0

            # Safety: fall back to uniform over legal moves if probs are bad
            if torch.isnan(probs).any() or probs.sum() < 1e-8:
                probs = legal_mask / legal_mask.sum()

            # Sample action
            action_idx = torch.multinomial(probs, num_samples=1).item()

            # Decode to a legal chess.Move
            chosen_move = index_to_move(action_idx, board)
            if chosen_move is None:
                chosen_move = next(iter(board.legal_moves))
                action_idx  = move_to_index(chosen_move, board.turn)

        # Store experience (value_target filled in later)
        history.append(
            {
                "state":  encode_board(board),   # CPU tensor
                "action": action_idx,
                "player": board.turn,            # chess.WHITE or chess.BLACK
            }
        )

        # Execute move
        pgn_node = pgn_node.add_variation(chosen_move)
        board.push(chosen_move)
        move_count += 1

    # ── Outcome ──────────────────────────────────────────────────────
    result = board.result(claim_draw=True)
    if move_count >= MAX_GAME_MOVES and not board.is_game_over(claim_draw=True):
        result = "1/2-1/2"
    pgn_game.headers["Result"] = result

    # Reward from white's point of view
    reward_white = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

    # Assign value targets from the perspective of the current player,
    # discounted backwards from the final move: vt = base_reward * (gamma ** steps_from_end)
    training_data: list[tuple[torch.Tensor, int, float]] = []
    total_steps = len(history)
    for t, entry in enumerate(history):
        base_vt = reward_white if entry["player"] == chess.WHITE else -reward_white
        # Decay factor: steps from end is (total_steps - 1 - t)
        decay = (gamma ** (total_steps - 1 - t)) if (gamma < 1.0 and base_vt != 0.0) else 1.0
        vt = base_vt * decay
        training_data.append((entry["state"], entry["action"], vt))

    # ── Export PGN ───────────────────────────────────────────────────
    os.makedirs(pgn_dir, exist_ok=True)
    # 1. Main cumulative log
    pgn_path = os.path.join(pgn_dir, "training_games.pgn")
    with open(pgn_path, "a", encoding="utf-8") as fh:
        print(pgn_game, file=fh, end="\n\n")

    # 2. 500-game interval logs with episode ranges
    training_sub_dir = os.path.join(pgn_dir, "training")
    os.makedirs(training_sub_dir, exist_ok=True)
    chunk_start = ((max(1, game_id) - 1) // 500) * 500 + 1
    chunk_end   = chunk_start + 499
    chunk_pgn_path = os.path.join(training_sub_dir, f"training_ep{chunk_start:05d}_to_ep{chunk_end:05d}.pgn")
    with open(chunk_pgn_path, "a", encoding="utf-8") as fh:
        print(pgn_game, file=fh, end="\n\n")

    return training_data, result, move_count
