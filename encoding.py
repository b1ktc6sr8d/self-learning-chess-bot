"""
Board state encoding and move mapping for the chess RL model.

The board is ALWAYS encoded from the perspective of the player whose turn it
is.  This means the neural network only needs to learn a single "role"
(the side to move) instead of learning separate strategies for white and
black.  When it is black's turn the board is vertically flipped so the
current player's pieces always appear on the bottom ranks.

Plane layout (18 × 8 × 8):
    0-5   : Current player's pieces  (P, N, B, R, Q, K)
    6-11  : Opponent's pieces        (P, N, B, R, Q, K)
    12    : Half-move clock          (normalised by 100, scalar fill)
    13    : Current player — kingside  castling rights  (0 or 1 fill)
    14    : Current player — queenside castling rights
    15    : Opponent — kingside  castling rights
    16    : Opponent — queenside castling rights
    17    : En-passant target square  (one-hot)

Action space:
    4 096 indices  →  from_square × 64 + to_square
    (Under-promotions share an index with queen promotions; queen
     promotion is always preferred when decoding.)
"""

from __future__ import annotations

import chess
import numpy as np
import torch

# ── Constants ────────────────────────────────────────────────────────────
PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN,  chess.KING,
]
NUM_PLANES  = 18
ACTION_SIZE = 64 * 64  # 4 096


# ── Helpers ──────────────────────────────────────────────────────────────
def _flip_square(sq: int) -> int:
    """Vertically mirror a square index (flip ranks, keep files)."""
    return sq ^ 56          # XOR with 0b111000 flips rank bits


# ── Board Encoding ───────────────────────────────────────────────────────
def encode_board(board: chess.Board) -> torch.Tensor:
    """Return an (18, 8, 8) float tensor from the current player's view."""
    planes   = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    is_white = board.turn == chess.WHITE

    # Pieces
    for i, pt in enumerate(PIECE_TYPES):
        for sq in board.pieces(pt, board.turn):            # my pieces → 0-5
            sq_e = sq if is_white else _flip_square(sq)
            r, c = divmod(sq_e, 8)
            planes[i, r, c] = 1.0

        for sq in board.pieces(pt, not board.turn):        # their pieces → 6-11
            sq_e = sq if is_white else _flip_square(sq)
            r, c = divmod(sq_e, 8)
            planes[i + 6, r, c] = 1.0

    # Half-move clock (50-move-rule proximity)
    planes[12, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    # Castling rights
    planes[13, :, :] = float(board.has_kingside_castling_rights(board.turn))
    planes[14, :, :] = float(board.has_queenside_castling_rights(board.turn))
    planes[15, :, :] = float(board.has_kingside_castling_rights(not board.turn))
    planes[16, :, :] = float(board.has_queenside_castling_rights(not board.turn))

    # En-passant
    if board.ep_square is not None:
        ep = board.ep_square if is_white else _flip_square(board.ep_square)
        r, c = divmod(ep, 8)
        planes[17, r, c] = 1.0

    return torch.from_numpy(planes)


# ── Move ↔ Index ─────────────────────────────────────────────────────────
def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """
    Map a ``chess.Move`` to an action index in [0, 4096).

    Squares are flipped when *turn* is black so the network always sees
    the board from the same orientation.
    """
    from_sq = move.from_square
    to_sq   = move.to_square
    if turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq   = _flip_square(to_sq)
    return from_sq * 64 + to_sq


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """
    Decode an action index into a legal ``chess.Move``.

    Un-flips squares for black, then scans ``board.legal_moves`` for a
    match.  Queen promotion is preferred when multiple promotions share
    the same index.  Returns ``None`` if nothing matches (safety net).
    """
    from_sq = index // 64
    to_sq   = index % 64
    if board.turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq   = _flip_square(to_sq)

    best: chess.Move | None = None
    for lm in board.legal_moves:
        if lm.from_square == from_sq and lm.to_square == to_sq:
            # Exact non-promotion or queen-promotion → return immediately
            if lm.promotion is None or lm.promotion == chess.QUEEN:
                return lm
            # Under-promotion → keep as fallback
            if best is None:
                best = lm
    return best


# ── Legal-Move Mask ──────────────────────────────────────────────────────
def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    """Return a float tensor of shape (4096,): 1 for legal actions, 0 else."""
    mask = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    for move in board.legal_moves:
        mask[move_to_index(move, board.turn)] = 1.0
    return mask
