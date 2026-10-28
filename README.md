# ♛ Chess RL — Pure Self-Play Reinforcement Learning Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-AlphaZero--Style%20ResNet-success)]()
[![Search](https://img.shields.io/badge/Search-2--Ply%20Batched%20Minimax-orange)]()
[![Games Played](https://img.shields.io/badge/Self--Play%20Games-2%2C252%20Recorded-brightgreen)]()

An autonomous chess intelligence developed **strictly from first principles (Tabula Rasa / Zero-Human-Data)** using deep reinforcement learning.

The model started with zero knowledge of chess strategy — **no opening books, no endgame tablebases, and no hardcoded piece values** (a Queen is not pre-assigned 9 points; a Pawn is not pre-assigned 1 point). All positional understanding, piece coordination, tactical traps, and king safety were discovered entirely through self-play games using a dual-head Residual Neural Network and a batched 2-ply Minimax lookahead engine.

---

## 📑 Table of Contents
- [Architecture & Neural Design](#-architecture--neural-design)
- [How It Learns (Reinforcement Learning Pipeline)](#-how-it-learns)
- [2-Ply Minimax Lookahead Engine](#-2-ply-minimax-lookahead-engine)
- [Self-Play Dataset & Training Milestones](#-self-play-dataset--training-milestones)
- [Quick Start & Installation](#-quick-start--installation)
- [Interactive Graphical Web Studio](#-interactive-graphical-web-studio)
- [Command-Line Play & Benchmarking](#-command-line-play--benchmarking)
- [Training & Resuming](#-training--resuming)
- [CLI Reference](#-cli-reference)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## 🧠 Architecture & Neural Design

The network is an **AlphaZero-style dual-head Residual Convolutional Neural Network (ResNet)** featuring ~1.74 million trainable parameters.

```
                         ┌─────────────────────────────────┐
                         │     18 × 8 × 8 Input Tensor     │
                         │   (Perspective-Encoded Board)   │
                         └────────────────┬────────────────┘
                                          │
                         ┌────────────────▼────────────────┐
                         │   Initial Conv (3×3, 128-F)     │
                         │       + BatchNorm + ReLU        │
                         └────────────────┬────────────────┘
                                          │
                         ┌────────────────▼────────────────┐
                         │     4 × Residual Blocks         │
                         │   ┌─────────────────────────┐   │
                         │   │ Conv 3×3 (128-F) + BN   │   │
                         │   │ ReLU                    │   │
                         │   │ Conv 3×3 (128-F) + BN   │   │
                         │   │ + Skip Connection       │   │
                         │   │ ReLU                    │   │
                         │   └─────────────────────────┘   │
                         └───────┬─────────────────┬───────┘
                                 │                 │
                ┌────────────────▼───┐  ┌──────────▼─────────────────┐
                │    Policy Head     │  │        Value Head          │
                │  Conv 1×1 (2-F)+BN │  │    Conv 1×1 (1-F) + BN     │
                │  Flatten → FC      │  │    Flatten → FC(128)       │
                │  Output: 4,096     │  │    ReLU → FC(1) → tanh     │
                │  (Action Logits)   │  │    Output: [-1.0, +1.0]    │
                └────────────────────┘  └───────────────────────────┘
```

### 1. Board Representation ($18 \times 8 \times 8$ Planes)
The board is represented from the **side-to-move's perspective** (the board is flipped symmetrically when Black is to move, halving the state space):
* **Planes 0–5:** Current player's pieces (`P`, `N`, `B`, `R`, `Q`, `K`).
* **Planes 6–11:** Opponent's pieces (`p`, `n`, `b`, `r`, `q`, `k`).
* **Planes 12–15:** Castling rights (White kingside, White queenside, Black kingside, Black queenside).
* **Plane 16:** En passant target square.
* **Plane 17:** Normalized half-move clock (50-move rule progress).

### 2. Action Space (4,096 Logits)
The discrete move space maps all 64 source squares to all 64 destination squares ($64 \times 64 = 4,096$).
* **python-chess is the sole referee:** The neural network never determines move legality. Legal move masks are applied to logits before softmax, setting illegal moves to $-\infty$.

---

## 🔬 How It Learns

The agent trains via **REINFORCE with a Learned Value Baseline and Discounted Advantage Actor-Critic**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} - c_{\text{entropy}} \cdot \mathcal{H}(\pi)$$

1. **Advantage-Weighted Policy Loss ($\mathcal{L}_{\text{policy}}$):**
   $$\mathcal{L}_{\text{policy}} = -\sum \log \pi(a_t | s_t) \cdot \left( R_t - V(s_t) \right)$$
   Weighting the policy gradient by the advantage $A(s_t) = R_t - V(s_t)$ dramatically reduces gradient variance.
2. **Value Prediction Loss ($\mathcal{L}_{\text{value}}$):**
   $$\mathcal{L}_{\text{value}} = \frac{1}{B} \sum \left( R_t - V(s_t) \right)^2$$
   The value head learns to predict game outcomes from raw board geometry.
3. **Entropy Regularization ($\mathcal{H}$):**
   Prevents premature policy collapse and encourages tactical exploration during early training.
4. **Reward Discount Factor ($\gamma = 0.99$):**
   Sparse terminal outcomes ($+1$ for win, $-1$ for loss, $0$ for draw) are discounted backwards:
   $$R_t = \text{outcome} \times \gamma^{(T - 1 - t)}$$
   This heavily incentivizes the bot to deliver swift, decisive checkmates rather than aimlessly shuffling pieces.

---

## ⚡ 2-Ply Minimax Lookahead Engine

To eliminate 0-ply tactical blindness (such as hanging pieces or falling for 1-move mating traps like Scholar's Mate), the engine incorporates a **2-ply lookahead search** with batched neural evaluation:

```
               Current Position S (Side to Move: C)
                                │
                   My Candidate Move m1 ∈ M1
                                │
                 Opponent Counter-Move m2 ∈ M2
                                │
                   Leaf Position S_{m1, m2}
                                │
             Neural Value Head: V(S_{m1, m2}) ∈ [-1, +1]
                                │
           Worst-Case Score: Q(m1) = min_{m2} V(S_{m1, m2})
                                │
       Optimal Move: argmax_{m1} [ Q(m1) + prior_weight · P(m1) ]
```

* **Vectorized Single-Pass Leaf Batching:** Rather than querying the neural network hundreds of times sequentially in a slow loop, all $m_1 \times m_2$ leaf states are stacked into **a single PyTorch tensor** and evaluated in one GPU/CPU forward pass.
* **Terminal Rules:** Immediate mates on Ply 1 score $+1.0$; opponent counter-mates on Ply 2 score $-1.0$; draws score $0.0$.
* **Perspective Invariance:** Because exactly 2 plies elapse (one by the player, one by the opponent), the turn at leaf $S_{m_1, m_2}$ returns to the original player, allowing direct evaluation without manual negation.

---

## 📊 Self-Play Dataset & Training Milestones

The engine has completed **2,252 self-play training games** alongside human matches and benchmark tests:

| Dataset / Archive | Path | Game Count | Description |
| :--- | :--- | :---: | :--- |
| **Training Epoch 1** | `games/training/training_ep00001_to_ep00500.pgn` | **702** | Initial exploration & basic piece development |
| **Training Epoch 2** | `games/training/training_ep00501_to_ep01000.pgn` | **500** | Center control discovery & early king safety |
| **Training Epoch 3** | `games/training/training_ep01001_to_ep01500.pgn` | **500** | Coordinated piece activity & tactical stability |
| **Training Epoch 4** | `games/training/training_ep01501_to_ep02000.pgn` | **500** | Mature 0-ply policy & endgame understanding |
| **Training Epoch 5** | `games/training/training_ep02001_to_ep02500.pgn` | **50** | 2-ply Minimax lookahead self-play fine-tuning |
| **Master Archive** | `games/training_games.pgn` | **2,252** | Complete cumulative training history |
| **Human Matches** | `games/human/*.pgn` | **7** | Real matches played against human players |
| **Benchmark Tests** | `games/benchmark/*.pgn` | **3** | Evaluative games vs random / automated benchmarks |
| **Grand Total** | | **2,262 Games** | **Complete Recorded Games in Dataset** |

Saved model checkpoints in `checkpoints/` allow inspecting the bot's intelligence progression from beginner (`ep00100`) to intermediate (`ep02000`).

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Reinforced-learning-based-Chess-Model.git
cd Reinforced-learning-based-Chess-Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requirements: `torch >= 2.0.0`, `chess >= 1.10.0`, `numpy >= 1.24.0`, `matplotlib >= 3.7.0`)*

### 3. Verify Installation
Run the automated test suite to confirm the model forward pass, encoding, and search pipeline:
```bash
python test_pipeline.py
```

---

## ♟️ Interactive Graphical Web Studio

Experience the bot in a web-based chess studio with zero external frontend dependencies:

```bash
python gui.py
```
Open your browser to `http://localhost:8080` to access:
* **Interactive SVG Chessboard:** Click-to-move, drag-and-drop, destination dots, and promotion modals.
* **Live Neural Evaluation Bar:** Displays real-time Value Head evaluation ($-1.0$ to $+1.0$) and win odds.
* **Live Search Thoughts:** Displays the top 3 candidate moves, anticipated opponent replies, and minimax $Q$-scores.
* **Checkpoint Selector:** Switch between any trained checkpoint (`model_ep00250.pt`, `model_ep02000.pt`, etc.) on the fly.
* **Automatic PGN Archiving:** Every completed game against a human is automatically saved to `games/human/` with full timestamps.

---

## 💻 Command-Line Play & Benchmarking

### Play as White or Black in the Terminal
```bash
# Play as White vs the 2-ply engine
python play.py --checkpoint checkpoints/model_ep02000.pt --color white --search 2ply

# Play as Black
python play.py --checkpoint checkpoints/model_ep02000.pt --color black --search 2ply
```

### Run a Benchmark (Bot vs. Random)
```bash
python play.py --checkpoint checkpoints/model_ep02000.pt --mode bot-vs-random --search 2ply
```

---

## 🏋️ Training & Resuming

### Resume Training with 2-Ply Lookahead
To continue training from Episode 2,000 using the 2-ply search engine:
```bash
python main.py --episodes 500 --resume checkpoints/model_ep02000.pt --search 2ply --max-candidates 16
```

### Start Fresh Training from Scratch
```bash
python main.py --episodes 500 --batch-size 256 --lr 1e-3 --gamma 0.99 --search 2ply
```

---

## ⚙️ CLI Reference

```bash
python main.py --help
```

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--episodes` | `500` | Number of self-play games to simulate |
| `--search` | `2ply` | Move selection mode: `2ply` (Minimax lookahead) or `0ply` (direct policy) |
| `--max-candidates` | `16` | Top candidate moves evaluated per ply in 2-ply search |
| `--batch-size` | `256` | Mini-batch size for neural network updates |
| `--lr` | `0.001` | Adam optimizer learning rate |
| `--gamma` | `0.99` | Reward discount factor ($0 < \gamma \le 1.0$) |
| `--entropy-coeff` | `0.01` | Policy entropy bonus coefficient in loss function |
| `--filters` | `128` | Number of convolutional filters in residual tower |
| `--res-blocks` | `4` | Number of residual blocks |
| `--temp-start` | `1.5` | Initial exploration temperature for move sampling |
| `--temp-end` | `0.4` | Final annealed temperature |
| `--resume` | `None` | Path to `.pt` checkpoint file to resume training from |
| `--pgn-dir` | `games` | Directory where PGN archives are written |
| `--checkpoint-dir`| `checkpoints`| Directory for saving model checkpoints |
| `--plot-dir` | `plots` | Directory for 4-panel training metrics PNGs |

---

## 📁 Repository Structure

```
.
├── encoding.py            # 18-plane board representation & 4,096-action mapping
├── model.py               # AlphaZero-style ResNet (4 ResBlocks, dual Policy/Value heads)
├── search.py              # 2-ply Minimax lookahead engine with batched leaf forward pass
├── self_play.py           # Self-play game simulation, reward assignment & PGN logging
├── train.py               # Replay buffer, REINFORCE training loop, and metrics plotting
├── main.py                # Main CLI entrypoint for training and resuming
├── play.py                # Terminal interactive play & benchmarking
├── gui.py                 # Self-contained browser-based Chess Studio with SVG pieces
├── test_pipeline.py       # End-to-end unit and smoke tests
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
├── .gitignore             # Standard Git ignore rules
├── checkpoints/           # Saved model weight checkpoints (.pt)
│   ├── model_ep00100.pt
│   ├── model_ep00250.pt
│   ├── ...
│   └── model_ep02000.pt
├── games/                 # Recorded PGN games
│   ├── training/          # Chunked training PGNs (every 500 episodes)
│   ├── human/             # Human vs Bot matches with timestamps
│   └── benchmark/         # Automated test matches
└── plots/                 # 4-panel training metric graphs (Value/Policy loss, Moves, etc.)
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — free for academic, personal, and commercial exploration.
