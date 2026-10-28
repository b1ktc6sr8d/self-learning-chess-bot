#!/usr/bin/env python3
"""
Entry point for the chess reinforcement-learning training system.

    python main.py                          # run with defaults (500 games)
    python main.py --episodes 2000 --lr 5e-4
    python main.py --help                   # see all options
"""

from __future__ import annotations

import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a chess engine from scratch via self-play RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Training ─────────────────────────────────────────────────────
    g = ap.add_argument_group("Training")
    g.add_argument("--episodes",         type=int,   default=500,
                   help="Total self-play games to generate")
    g.add_argument("--batch-size",       type=int,   default=256,
                   help="Mini-batch size for each gradient step")
    g.add_argument("--lr",               type=float, default=1e-3,
                   help="Adam learning rate")
    g.add_argument("--games-per-update", type=int,   default=5,
                   help="Self-play games between weight updates")
    g.add_argument("--train-steps",      type=int,   default=4,
                   help="Gradient steps per update window")

    # ── Architecture ─────────────────────────────────────────────────
    g = ap.add_argument_group("Model architecture")
    g.add_argument("--filters",    type=int, default=128,
                   help="Convolutional filter count in residual tower")
    g.add_argument("--res-blocks", type=int, default=4,
                   help="Number of residual blocks")

    # ── Exploration ──────────────────────────────────────────────────
    g = ap.add_argument_group("Exploration")
    g.add_argument("--temp-start",    type=float, default=1.5,
                   help="Initial softmax temperature (high = exploratory)")
    g.add_argument("--temp-end",      type=float, default=0.4,
                   help="Final softmax temperature (low = exploitative)")
    g.add_argument("--temp-decay",    type=int,   default=300,
                   help="Episodes over which temperature anneals linearly")
    g.add_argument("--entropy-coeff", type=float, default=0.01,
                   help="Entropy bonus coefficient in the loss")
    g.add_argument("--gamma",         type=float, default=0.99,
                   help="Reward discount factor (0 < gamma <= 1.0) to encourage faster wins")

    # ── Search & Lookahead ───────────────────────────────────────────
    g = ap.add_argument_group("Search & Lookahead")
    g.add_argument("--search",         choices=["2ply", "0ply"], default="2ply",
                   help="Search depth: 2-ply Minimax lookahead or 0-ply direct policy")
    g.add_argument("--max-candidates", type=int, default=16,
                   help="Maximum candidate moves evaluated per ply in 2-ply search")

    # ── Storage ──────────────────────────────────────────────────────
    g = ap.add_argument_group("Storage")
    g.add_argument("--pgn-dir",       type=str, default="games",
                   help="Directory for PGN game records")
    g.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory for model checkpoints")
    g.add_argument("--checkpoint-interval", type=int, default=100,
                   help="Save a checkpoint every N episodes")
    g.add_argument("--buffer-capacity", type=int, default=100_000,
                   help="Maximum replay-buffer size (positions)")
    g.add_argument("--resume",         type=str, default=None,
                   help="Path to checkpoint .pt file to resume training from")
    g.add_argument("--plot-dir",       type=str, default="plots",
                   help="Directory for metrics PNG graphs")
    g.add_argument("--plot-interval",  type=int, default=500,
                   help="Generate metrics PNG plot every N episodes")

    args = ap.parse_args()

    config = {
        # Training
        "num_episodes":          args.episodes,
        "batch_size":            args.batch_size,
        "lr":                    args.lr,
        "weight_decay":          1e-4,
        "games_per_update":      args.games_per_update,
        "train_steps_per_update": args.train_steps,
        # Architecture
        "num_filters":           args.filters,
        "num_res_blocks":        args.res_blocks,
        # Exploration & Rewards
        "temp_start":            args.temp_start,
        "temp_end":              args.temp_end,
        "temp_decay_episodes":   args.temp_decay,
        "entropy_coeff":         args.entropy_coeff,
        "gamma":                 args.gamma,
        # Search & Lookahead
        "use_2ply":              args.search == "2ply",
        "max_candidates":        args.max_candidates,
        # Storage & Visualizations
        "pgn_dir":               args.pgn_dir,
        "checkpoint_dir":        args.checkpoint_dir,
        "checkpoint_interval":   args.checkpoint_interval,
        "buffer_capacity":       args.buffer_capacity,
        "resume":                args.resume,
        "plot_dir":              args.plot_dir,
        "plot_interval":         args.plot_interval,
    }

    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
