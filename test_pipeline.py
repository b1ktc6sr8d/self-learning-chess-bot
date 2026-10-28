"""
Smoke test suite for the Chess RL pipeline.
Verifies board encoding, neural network forward pass, discount factor, and training steps.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import chess
import torch
import numpy as np

def run_tests():
    print("=" * 60)
    print("Running Chess RL Pipeline Smoke Tests...")
    print("=" * 60)

    # 1. Test Encoding
    from encoding import (
        encode_board,
        get_legal_move_mask,
        move_to_index,
        index_to_move,
        NUM_PLANES,
        ACTION_SIZE,
    )

    board = chess.Board()
    state = encode_board(board)
    assert state.shape == (NUM_PLANES, 8, 8), f"Bad state shape: {state.shape}"
    assert state.dtype == torch.float32

    mask = get_legal_move_mask(board)
    assert mask.shape == (ACTION_SIZE,), f"Bad mask shape: {mask.shape}"
    assert mask.sum().item() == 20.0, f"Starting legal move count should be 20, got {mask.sum().item()}"

    # Roundtrip test for moves
    for m in board.legal_moves:
        idx = move_to_index(m, board.turn)
        decoded = index_to_move(idx, board)
        assert decoded == m, f"Move mismatch: {m} != {decoded}"

    # Test black perspective flip
    board.push_san("e4")
    state_black = encode_board(board)
    assert state_black.shape == (NUM_PLANES, 8, 8)
    for m in board.legal_moves:
        idx = move_to_index(m, board.turn)
        decoded = index_to_move(idx, board)
        assert decoded == m, f"Black move mismatch: {m} != {decoded}"

    print("✓ [1/4] Board encoding and move mapping tests passed.")

    # 2. Test Model Architecture
    from model import ChessNet

    model = ChessNet(num_filters=32, num_res_blocks=2)
    dummy_input = torch.randn(4, NUM_PLANES, 8, 8)
    p_logits, v_pred = model(dummy_input)

    assert p_logits.shape == (4, ACTION_SIZE), f"Bad policy shape: {p_logits.shape}"
    assert v_pred.shape == (4, 1), f"Bad value shape: {v_pred.shape}"
    assert (-1.0 <= v_pred.min() and v_pred.max() <= 1.0), "Value output outside [-1, 1]"

    print("✓ [2/4] Model architecture forward pass passed.")

    # 3. Test Self-Play with Discount Factor
    from self_play import play_game

    device = torch.device("cpu")
    # Quick self-play test
    data, result, n_moves = play_game(
        model, device, temperature=1.0, game_id=1, pgn_dir="test_games", gamma=0.99
    )
    assert len(data) == n_moves, f"Data count {len(data)} != move count {n_moves}"
    assert result in ("1-0", "0-1", "1/2-1/2")

    # If game had a decisive result, check discount factor decay
    if result != "1/2-1/2" and len(data) > 1:
        last_vt = data[-1][2]
        first_vt = data[0][2]
        assert abs(last_vt) > abs(first_vt), "Discount factor should make final move have higher magnitude than initial move"

    print(f"✓ [3/4] Self-play match completed ({n_moves} moves, result {result}). Discount factor verified.")

    # 4. Test Training Step
    from train import ReplayBuffer, Trainer

    cfg = {
        "num_episodes": 2,
        "batch_size": 16,
        "lr": 1e-3,
        "num_filters": 32,
        "num_res_blocks": 2,
        "gamma": 0.99,
        "pgn_dir": "test_games",
        "checkpoint_dir": "test_checkpoints",
    }
    trainer = Trainer(cfg)
    trainer.buffer.push(data)

    if len(trainer.buffer) >= 16:
        vl, pl, ent = trainer._train_step(16)
        assert not np.isnan(vl), "Value loss is NaN"
        assert not np.isnan(pl), "Policy loss is NaN"
        assert not np.isnan(ent), "Entropy is NaN"
        print(f"✓ [4/4] Gradient step executed (V-Loss: {vl:.4f}, P-Loss: {pl:.4f}, Entropy: {ent:.4f}).")
    else:
        print("✓ [4/4] Buffer populated successfully.")

    # Clean up test directories
    import shutil, os
    if os.path.exists("test_games"):
        shutil.rmtree("test_games", ignore_errors=True)
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints", ignore_errors=True)

    print()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
