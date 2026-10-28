"""
2-Ply Minimax Lookahead Engine with Batched Neural Value Evaluation.

Architecture:
    Current State S (Player to move: C)
       │
       ├── My Candidate Move m1 ∈ M1
       │      │
       │      ├── Opponent Counter-Move m2 ∈ M2
       │      │      └── Leaf State S_{m1, m2} (Side to move: C)
       │      │             └── Evaluated via Neural Value Head V(S_{m1, m2}) ∈ [-1, +1]
       │      │
       │      └── Worst-Case Score: Q(m1) = min_{m2} Score(S_{m1, m2})
       │
       └── Optimal Move: argmax_{m1} [ Q(m1) + prior_weight * P(m1) ]

All leaf states across all candidate branches are batched into a single PyTorch
forward pass, eliminating per-node overhead and maximizing GPU/CPU vectorization.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch

from encoding import (
    ACTION_SIZE,
    encode_board,
    get_legal_move_mask,
    index_to_move,
    move_to_index,
)

if TYPE_CHECKING:
    from model import ChessNet


def select_move_2ply(
    model: ChessNet,
    board: chess.Board,
    device: torch.device,
    temperature: float = 0.0,
    max_m1: int | None = 20,
    max_m2: int | None = 20,
    prior_weight: float = 0.15,
) -> tuple[chess.Move, int, float, list[dict]]:
    """
    Select the optimal legal move using 2-ply Minimax with batched neural evaluation.

    Parameters
    ----------
    model : ChessNet
        Dual-head neural network (Policy + Value).
    board : chess.Board
        Current board state.
    device : torch.device
        Inference device (CPU or CUDA).
    temperature : float
        Softmax temperature (0.0 = pure greedy minimax, >0 = exploratory sampling).
    max_m1 : int | None
        Maximum candidate moves evaluated for current player (None = all legal).
    max_m2 : int | None
        Maximum counter-moves simulated for opponent (None = all legal).
    prior_weight : float
        Influence of policy head prior in combined scoring (Score = Q + prior_weight * P).

    Returns
    -------
    chosen_move : chess.Move
        The best legal move selected.
    action_idx : int
        Action index in [0, 4096) for training replay buffer.
    best_value : float
        Minimax guaranteed value score in [-1.0, +1.0].
    candidate_details : list[dict]
        Top candidate moves with their minimax Q-score, prior, and combined score.
    """
    legal_m1 = list(board.legal_moves)
    if not legal_m1:
        raise ValueError("Cannot search on a terminal board with no legal moves.")

    # Trivial case: only 1 legal move (e.g. forced response)
    if len(legal_m1) == 1:
        m = legal_m1[0]
        idx = move_to_index(m, board.turn)
        return m, idx, 0.0, [{"san": board.san(m), "q": 0.0, "prob": 100.0, "uci": m.uci()}]

    model.eval()

    # ── Step 1: Root Policy & Value evaluation ────────────────────────
    root_tensor = encode_board(board).unsqueeze(0).to(device)
    legal_mask = get_legal_move_mask(board).to(device)

    with torch.no_grad():
        policy_logits, root_val_tensor = model(root_tensor)

    policy_logits = policy_logits.squeeze(0)
    policy_logits[legal_mask == 0] = float("-inf")
    root_probs = torch.softmax(policy_logits, dim=0).cpu().numpy()

    # Map each legal move to its policy prior probability
    priors: dict[chess.Move, float] = {}
    for m in legal_m1:
        idx = move_to_index(m, board.turn)
        priors[m] = float(root_probs[idx])

    # ── Step 2: Candidate move selection for Ply 1 ────────────────────
    # Prioritize moves by policy prior, captures, and checks
    if max_m1 is not None and len(legal_m1) > max_m1:
        def move_priority(m: chess.Move) -> float:
            score = priors.get(m, 0.0)
            if board.is_capture(m):
                score += 0.25
            if board.gives_check(m):
                score += 0.20
            return score

        candidates_m1 = sorted(legal_m1, key=move_priority, reverse=True)[:max_m1]
    else:
        candidates_m1 = legal_m1

    # ── Step 3: Expand 2-ply Tree & Collect Leaf States ───────────────
    # We will populate leaf_states for non-terminal 2-ply leaves
    # leaf_scores maps (m1, m2) -> float score from player's perspective
    leaf_scores: dict[tuple[chess.Move, chess.Move], float] = {}
    leaf_states: list[torch.Tensor] = []
    leaf_keys: list[tuple[chess.Move, chess.Move]] = []
    immediate_m1_scores: dict[chess.Move, float] = {}

    for m1 in candidates_m1:
        board.push(m1)

        # Terminal check at Ply 1
        if board.is_checkmate():
            # Current player delivers immediate checkmate!
            immediate_m1_scores[m1] = 1.0
            board.pop()
            continue
        elif board.is_stalemate() or board.is_insufficient_material():
            immediate_m1_scores[m1] = 0.0
            board.pop()
            continue

        # Opponent's counter-moves at Ply 2
        legal_m2 = list(board.legal_moves)
        if not legal_m2:
            immediate_m1_scores[m1] = 0.0
            board.pop()
            continue

        # Optional pruning of opponent counter-moves
        if max_m2 is not None and len(legal_m2) > max_m2:
            # Opponent captures & checks are most critical to examine
            def opp_priority(m: chess.Move) -> float:
                p = 0.0
                if board.is_capture(m):
                    p += 1.0
                if board.gives_check(m):
                    p += 0.8
                return p

            candidates_m2 = sorted(legal_m2, key=opp_priority, reverse=True)[:max_m2]
        else:
            candidates_m2 = legal_m2

        for m2 in candidates_m2:
            board.push(m2)

            # Terminal check at Ply 2
            if board.is_checkmate():
                # Opponent delivers checkmate on ply 2 -> catastrophic for current player
                leaf_scores[(m1, m2)] = -1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                leaf_scores[(m1, m2)] = 0.0
            else:
                # Non-terminal leaf: side-to-move is current player again!
                # encode_board naturally encodes from the current player's perspective.
                leaf_states.append(encode_board(board))
                leaf_keys.append((m1, m2))

            board.pop()  # pop m2

        board.pop()  # pop m1

    # ── Step 4: Batched Neural Value Head Inference ───────────────────
    if leaf_states:
        # Single forward pass for all leaves across all candidate branches!
        leaf_batch = torch.stack(leaf_states).to(device)
        with torch.no_grad():
            _, leaf_values = model(leaf_batch)
        leaf_vals = leaf_values.squeeze(-1).cpu().numpy()

        for (m1, m2), val in zip(leaf_keys, leaf_vals):
            leaf_scores[(m1, m2)] = float(val)

    # ── Step 5: Minimax Reduction for each m1 ─────────────────────────
    # Q(m1) is the worst-case score after the opponent's best response
    q_scores: dict[chess.Move, float] = {}

    for m1 in candidates_m1:
        if m1 in immediate_m1_scores:
            q_scores[m1] = immediate_m1_scores[m1]
        else:
            # Find all m2 responses evaluated for this m1
            branch_scores = [score for (branch_m1, _), score in leaf_scores.items() if branch_m1 == m1]
            if branch_scores:
                # Opponent minimizes player's outcome
                q_scores[m1] = min(branch_scores)
            else:
                q_scores[m1] = 0.0

    # ── Step 6: Combined Scoring & Decision Rule ──────────────────────
    combined_scores: dict[chess.Move, float] = {}
    details: list[dict] = []

    for m1 in candidates_m1:
        q = q_scores[m1]
        prior = priors.get(m1, 0.0)
        # Combined score incorporates minimax guarantee + policy prior bonus
        score = q + prior_weight * prior
        combined_scores[m1] = score
        details.append({
            "move": m1,
            "uci": m1.uci(),
            "san": board.san(m1),
            "q": round(q, 3),
            "prior": round(prior * 100, 1),
            "score": round(score, 3),
        })

    # Sort details by combined score descending
    details.sort(key=lambda d: d["score"], reverse=True)

    if temperature <= 0.01:
        # Greedy selection: highest combined minimax score
        chosen_move = details[0]["move"]
        best_q = details[0]["q"]
    else:
        # Exploratory temperature sampling over top candidates
        cand_moves = [d["move"] for d in details]
        cand_scores = np.array([d["score"] for d in details], dtype=np.float32)
        # Softmax over temperature
        cand_scores = cand_scores - np.max(cand_scores)  # stability
        exp_scores = np.exp(cand_scores / temperature)
        probs = exp_scores / (np.sum(exp_scores) + 1e-8)
        chosen_idx = np.random.choice(len(cand_moves), p=probs)
        chosen_move = cand_moves[chosen_idx]
        best_q = q_scores[chosen_move]

    action_idx = move_to_index(chosen_move, board.turn)
    return chosen_move, action_idx, best_q, details
