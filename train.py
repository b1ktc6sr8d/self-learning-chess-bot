"""
Training pipeline for the self-play chess RL system.

Learning algorithm — **REINFORCE with a learned value baseline**:

    Policy loss   =  −log π(a|s) · A(s)          (advantage-weighted)
    Value  loss   =  (R − V(s))²                  (mean-squared error)
    Entropy bonus =  −Σ π log π                    (discourages premature collapse)

    where  A(s) = R − V(s)     (advantage)
           R    ∈ {+1, 0, −1}  (sparse game outcome)

No hardcoded piece values, opening books, or endgame tables.
All knowledge is acquired through self-play.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ChessNet
from self_play import play_game


# ── Metrics Plotting ─────────────────────────────────────────────────────

def save_training_plots(history: dict, save_path: str) -> None:
    """
    Generate a 4-panel PNG dashboard tracking training metrics:
    1. Value Loss (Prediction Error)
    2. Policy Loss (REINFORCE)
    3. Game Length Efficiency (Moves per Game)
    4. Win / Loss / Draw Outcome Distribution
    """
    if not history["episodes"]:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    current_ep = history["episodes"][-1]
    fig.suptitle(f"♛ Chess RL — Training Metrics (Episode {current_ep})", fontsize=16, fontweight="bold")

    # 1. Value Loss (MSE)
    ax1 = axes[0, 0]
    ax1.plot(history["episodes"], history["value_losses"], color="#e74c3c", lw=2.0, label="Value Loss (MSE)")
    ax1.set_title("Value Head Error (Win Prediction Loss)", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Squared Error")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. Policy Loss
    ax2 = axes[0, 1]
    ax2.plot(history["episodes"], history["policy_losses"], color="#3498db", lw=2.0, label="Policy Loss")
    ax2.set_title("Policy Head Loss (REINFORCE + Advantage)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 3. Game Length Efficiency
    ax3 = axes[1, 0]
    ax3.plot(history["episodes"], history["raw_moves"], color="#bdc3c7", alpha=0.4, label="Game Moves (Half-moves)")
    ax3.plot(history["episodes"], history["avg_moves"], color="#2ecc71", lw=2.0, label="Rolling Avg Moves")
    ax3.set_title("Game Length / Tempo Efficiency", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Half-Moves")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # 4. Outcome Distribution
    ax4 = axes[1, 1]
    total_games = max(1, history["white_wins"] + history["black_wins"] + history["draws"])
    w_pct = history["white_wins"] / total_games * 100
    b_pct = history["black_wins"] / total_games * 100
    d_pct = history["draws"] / total_games * 100

    categories = ["White Wins\n(1-0)", "Black Wins\n(0-1)", "Draws\n(1/2-1/2)"]
    percentages = [w_pct, b_pct, d_pct]
    colors = ["#f39c12", "#34495e", "#7f8c8d"]
    bars = ax4.bar(categories, percentages, color=colors, width=0.55)
    ax4.set_title(f"Cumulative Outcome Distribution ({total_games} Games)", fontweight="bold")
    ax4.set_ylabel("Percentage (%)")
    ax4.set_ylim(0, 105)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   📊 Metrics plot saved → {save_path}")


# ── Replay Buffer ────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity ring buffer of (state, action, value_target) tuples."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buf: deque[tuple[torch.Tensor, int, float]] = deque(maxlen=capacity)

    def push(self, data: list[tuple[torch.Tensor, int, float]]) -> None:
        self._buf.extend(data)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n   = min(batch_size, len(self._buf))
        idx = np.random.choice(len(self._buf), size=n, replace=False)
        batch = [self._buf[i] for i in idx]
        states  = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        values  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        return states, actions, values

    def __len__(self) -> int:
        return len(self._buf)


# ── Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates self-play data generation and neural-network training."""

    def __init__(self, config: dict) -> None:
        self.cfg    = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = ChessNet(
            num_filters    = config.get("num_filters", 128),
            num_res_blocks = config.get("num_res_blocks", 4),
        ).to(self.device)

        # Optimiser
        self.optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr           = config.get("lr", 1e-3),
            weight_decay = config.get("weight_decay", 1e-4),
        )

        # Experience replay
        self.buffer = ReplayBuffer(config.get("buffer_capacity", 100_000))

        # Output directories
        self.pgn_dir       = config.get("pgn_dir", "games")
        self.ckpt_dir      = config.get("checkpoint_dir", "checkpoints")
        self.plot_dir      = config.get("plot_dir", "plots")
        self.plot_interval = config.get("plot_interval", 500)
        os.makedirs(self.pgn_dir,  exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Rolling metrics (window = last 100 updates / games)
        self.value_losses  = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.entropies     = deque(maxlen=100)
        self.game_lengths  = deque(maxlen=100)
        self.results: dict[str, int] = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        # Full metric history for plotting
        self.history: dict = {
            "episodes": [],
            "value_losses": [],
            "policy_losses": [],
            "raw_moves": [],
            "avg_moves": [],
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }

        # Resume from checkpoint if specified
        self.start_episode = 0
        resume_path = config.get("resume")
        if resume_path:
            if os.path.isfile(resume_path):
                ckpt = torch.load(resume_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    self.optimiser.load_state_dict(ckpt["optimizer_state_dict"])
                self.start_episode = ckpt.get("episode", 0)
                if "results" in ckpt:
                    self.results.update(ckpt["results"])
                if "history" in ckpt:
                    self.history = ckpt["history"]
                print(f"   ✓ Resumed from checkpoint: {resume_path} (Starting at ep {self.start_episode + 1})")
            else:
                print(f"   ⚠ Checkpoint not found at '{resume_path}'. Starting from scratch.")

    # ── Helpers ──────────────────────────────────────────────────────

    def _temperature(self, episode: int) -> float:
        """Linear annealing from high exploration to low exploitation."""
        t0    = self.cfg.get("temp_start", 1.5)
        t1    = self.cfg.get("temp_end",   0.4)
        decay = self.cfg.get("temp_decay_episodes", 300)
        return t0 + (t1 - t0) * min(episode / decay, 1.0)

    # ── Single Gradient Step ─────────────────────────────────────────

    def _train_step(self, batch_size: int) -> tuple[float, float, float]:
        self.model.train()

        states, actions, value_targets = self.buffer.sample(batch_size)
        states        = states.to(self.device)
        actions       = actions.to(self.device)
        value_targets = value_targets.to(self.device)

        policy_logits, value_preds = self.model(states)
        value_preds = value_preds.squeeze(-1)              # (B,)

        # ── Value loss (MSE) ─────────────────────────────────────
        value_loss = F.mse_loss(value_preds, value_targets)

        # ── Policy loss (REINFORCE with value baseline) ──────────
        advantages       = value_targets - value_preds.detach()
        log_probs        = F.log_softmax(policy_logits, dim=1)
        selected_lp      = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss      = -(selected_lp * advantages).mean()

        # ── Entropy bonus ────────────────────────────────────────
        probs        = F.softmax(policy_logits, dim=1)
        entropy      = -(probs * log_probs).sum(dim=1).mean()
        entropy_coef = self.cfg.get("entropy_coeff", 0.01)

        # ── Backward ─────────────────────────────────────────────
        total_loss = policy_loss + value_loss - entropy_coef * entropy

        self.optimiser.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()

        return value_loss.item(), policy_loss.item(), entropy.item()

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self) -> ChessNet:
        """Execute the full self-play → train loop and return the model."""
        num_episodes       = self.cfg.get("num_episodes", 500)
        batch_size         = self.cfg.get("batch_size", 256)
        games_per_update   = self.cfg.get("games_per_update", 5)
        steps_per_update   = self.cfg.get("train_steps_per_update", 4)
        ckpt_interval      = self.cfg.get("checkpoint_interval", 100)

        param_count = sum(p.numel() for p in self.model.parameters())

        # ── Banner ───────────────────────────────────────────────
        print("=" * 76)
        print("   ♛  CHESS RL  —  PURE SELF-PLAY TRAINING  ♛")
        print("=" * 76)
        print(f"   Device              {self.device}")
        print(f"   Model parameters    {param_count:,}")
        print(f"   Residual blocks     {self.cfg.get('num_res_blocks', 4)}")
        print(f"   Filters             {self.cfg.get('num_filters', 128)}")
        print(f"   Search engine       {'2-ply Minimax (batched)' if self.cfg.get('use_2ply', True) else '0-ply Direct Policy'}")
        if self.cfg.get("use_2ply", True):
            print(f"   Max candidates/ply  {self.cfg.get('max_candidates', 16)}")
        print(f"   Episodes            {num_episodes}" + (f" (Resumed from {self.start_episode})" if self.start_episode > 0 else ""))
        print(f"   Batch size          {batch_size}")
        print(f"   Games / update      {games_per_update}")
        print(f"   Train steps / upd   {steps_per_update}")
        print(f"   Learning rate       {self.cfg.get('lr', 1e-3)}")
        print(f"   Discount factor     {self.cfg.get('gamma', 0.99)}")
        print(f"   Entropy coeff       {self.cfg.get('entropy_coeff', 0.01)}")
        print(f"   PGN output          {os.path.abspath(self.pgn_dir)}")
        print(f"   Checkpoints         {os.path.abspath(self.ckpt_dir)}")
        print(f"   Plot output         {os.path.abspath(self.plot_dir)} (every {self.plot_interval} eps)")
        print("=" * 76)
        print()

        t0 = time.time()
        gamma = self.cfg.get("gamma", 0.99)
        use_2ply = self.cfg.get("use_2ply", True)
        max_candidates = self.cfg.get("max_candidates", 16)
        total_target_episodes = self.start_episode + num_episodes

        for ep in range(self.start_episode + 1, total_target_episodes + 1):
            temp = self._temperature(ep)

            # ── Self-play ────────────────────────────────────────
            data, result, n_moves = play_game(
                self.model, self.device,
                temperature=temp, game_id=ep, pgn_dir=self.pgn_dir,
                gamma=gamma,
                use_2ply=use_2ply,
                max_candidates=max_candidates,
            )
            self.buffer.push(data)
            self.game_lengths.append(n_moves)
            self.results[result] = self.results.get(result, 0) + 1

            if result == "1-0":
                self.history["white_wins"] += 1
            elif result == "0-1":
                self.history["black_wins"] += 1
            else:
                self.history["draws"] += 1

            # ── Training ─────────────────────────────────────────
            if ep % games_per_update == 0 and len(self.buffer) >= batch_size:
                for _ in range(steps_per_update):
                    vl, pl, ent = self._train_step(batch_size)
                    self.value_losses.append(vl)
                    self.policy_losses.append(pl)
                    self.entropies.append(ent)

                avg_vl  = np.mean(self.value_losses)
                avg_pl  = np.mean(self.policy_losses)
                avg_len = np.mean(self.game_lengths)
                elapsed = time.time() - t0

                # Record history for plotting
                self.history["episodes"].append(ep)
                self.history["value_losses"].append(float(avg_vl))
                self.history["policy_losses"].append(float(avg_pl))
                self.history["raw_moves"].append(n_moves)
                self.history["avg_moves"].append(float(avg_len))

                print(
                    f"Ep {ep:5d}/{num_episodes} │ "
                    f"Result {result:7s} │ "
                    f"Moves {n_moves:4d} │ "
                    f"Avg Moves {avg_len:6.1f} │ "
                    f"V-Loss {avg_vl:.4f} │ "
                    f"P-Loss {avg_pl:.4f} │ "
                    f"Temp {temp:.2f} │ "
                    f"Buf {len(self.buffer):6d} │ "
                    f"{elapsed:6.0f}s"
                )
            else:
                # Lightweight log for non-training episodes
                if ep <= 20 or ep % 10 == 0:
                    print(
                        f"Ep {ep:5d}/{num_episodes} │ "
                        f"Result {result:7s} │ "
                        f"Moves {n_moves:4d} │ "
                        f"Temp {temp:.2f}"
                    )

            # ── Metric Plots ─────────────────────────────────────
            if ep % self.plot_interval == 0 and len(self.history["episodes"]) > 0:
                plot_path = os.path.join(self.plot_dir, f"metrics_ep{ep:05d}.png")
                save_training_plots(self.history, plot_path)

            # ── Checkpoint ───────────────────────────────────────
            if ep % ckpt_interval == 0:
                path = os.path.join(self.ckpt_dir, f"model_ep{ep:05d}.pt")
                torch.save(
                    {
                        "episode":              ep,
                        "model_state_dict":     self.model.state_dict(),
                        "optimizer_state_dict":  self.optimiser.state_dict(),
                        "results":              dict(self.results),
                        "history":              dict(self.history),
                    },
                    path,
                )
                print(f"   ✓ Checkpoint saved → {path}")

        # ── Final report & Final Plot ─────────────────────────────
        elapsed = time.time() - t0
        print()
        print("=" * 76)
        print("   TRAINING COMPLETE")
        print("=" * 76)
        print(f"   Wall time         {elapsed:,.1f}s  ({elapsed / 60:.1f} min)")
        print(f"   Games played      {num_episodes}")
        print(f"   White wins        {self.results.get('1-0', 0)}")
        print(f"   Black wins        {self.results.get('0-1', 0)}")
        print(f"   Draws             {self.results.get('1/2-1/2', 0)}")
        if self.value_losses:
            print(f"   Final Value Loss  {np.mean(self.value_losses):.4f}")
            print(f"   Final Policy Loss {np.mean(self.policy_losses):.4f}")
        if self.game_lengths:
            print(f"   Avg game length   {np.mean(self.game_lengths):.1f} half-moves")
        print("=" * 76)

        if len(self.history["episodes"]) > 0:
            final_plot = os.path.join(self.plot_dir, f"metrics_final_ep{total_target_episodes:05d}.png")
            save_training_plots(self.history, final_plot)

        return self.model
