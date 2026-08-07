# RL Final Project — Intelligent Agent in a Dynamic Maze

**Student ID:** 40406934 | **Seed:** 3 | **Maze size:** 18 × 18

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment & MDP](#environment--mdp)
3. [Algorithms](#algorithms)
4. [Transfer Learning](#transfer-learning)
5. [GUI](#gui)
6. [Installation](#installation)
7. [Config identity](#config-identity)
8. [Running the Project](#running-the-project)
9. [Reproducing Results](#reproducing-results-phases-4-9)
10. [Project Structure](#project-structure)
11. [Results Summary](#results-summary)
12. [AI Usage Disclosure](#ai-usage-disclosure)
13. [References](#references)

---

## Project Overview

<!-- TODO: fill in after implementation -->

## Environment & MDP

<!-- TODO: describe maze, state space (x, y, has_key, energy), action space,
     transition function (0.8/0.1/0.1), reward versions, episode termination -->

## Algorithms

### Value Iteration

<!-- TODO: Bellman backup, convergence criterion, γ sweep -->

### Q-Learning

<!-- TODO: off-policy, ε-greedy, linear vs exponential decay -->

### SARSA(λ)

<!-- TODO: on-policy, replacing traces, λ sweep -->

### Cross-Algorithm Comparison

<!-- TODO: runtime, sample efficiency, policy agreement with VI -->

## Transfer Learning

<!-- TODO: source env, similar target, different target, 4 scenarios, negative transfer case -->

## GUI

<!-- TODO: controls, overlays, visual elements -->

## Installation

```bash
git clone https://github.com/D3rhami/RL_FinalProject_40406934.git
cd RL_FinalProject_40406934
pip install -r requirements.txt
```

Run all commands below from the repository root.

`requirements.txt` pins: `numpy`, `pandas`, `matplotlib`, `customtkinter`,
`pillow`, `pytest`, and `pygame` (optional animation; GUI uses customtkinter).

## Config identity

From `experiments/configs/default_config.json` (also shown in the header above):

| Field | Value |
|-------|-------|
| `student_id` | `40406934` |
| `base_seed` | `3` |
| `maze_size` | `18` (18 × 18) |
| `env.max_energy` | `60` |

Training episode counts in the same config:

- Q-Learning / SARSA(λ): **50,000** episodes (`q_learning.num_episodes`, `sarsa_lambda.num_episodes`)
- Transfer: **shaped** reward, **5,000** episodes (`transfer.reward_mode`, `transfer.num_episodes`)

## Running the Project

### Maps (generate if needed)

```bash
# Source maze → environments/maps/source_maze.json
python environments/generator.py

# Transfer targets → target_similar.json / target_different.json
python environments/generator.py --targets
```

Skip these if `environments/maps/*.json` is already present; re-run only to regenerate.

### Experiments

```bash
# Full default grid: VI + Q-Learning + SARSA (transfer is separate)
python experiments/run_experiments.py

# Selective: vi | q_learning | sarsa | transfer
python experiments/run_experiments.py vi
python experiments/run_experiments.py q_learning
python experiments/run_experiments.py sarsa
python experiments/run_experiments.py transfer
```

Aliases accepted: `vi` / `value_iteration`, `q_learning`, `sarsa` / `sarsa_lambda`,
`transfer` / `transfer_learning`. Multiple names can be passed on one line
(e.g. `python experiments/run_experiments.py q_learning sarsa`).

Optional flags: `--config PATH`, `--dry-run` (short episodes for smoke tests),
`--fresh` (clear `results/raw_data/run_ledger.csv` before starting).

QL/SARSA train for **50,000** episodes; transfer uses **shaped** reward and
**5,000** episodes (see Config identity).

### Figures

```bash
python experiments/analysis.py && python experiments/compare.py
```

### GUI

```bash
python gui/app.py
# or
python -m gui.app
```

### Tests

```bash
pytest tests/
```

### Single-algorithm CLI (smoke / one-off)

```bash
python main.py --algo value_iteration --gamma 0.95 --max-iterations 1000
python main.py --algo q_learning --schedule linear --num-episodes 10000
python main.py --algo q_learning --schedule exponential --num-episodes 10000
python main.py --algo sarsa_lambda --lam 0.7 --num-episodes 10000
```

`main.py --algo …` writes a single model JSON (+ a summary CSV for QL/SARSA)
compatible with `results/models/{vi,q_learning,sarsa}/`. It does **not** produce
the full multi-seed / multi-variant CSVs that `analysis.py` and `compare.py`
expect — use `experiments/run_experiments.py` for that.

### Energy sweep

```bash
python experiments/energy_sweep.py
```

## Reproducing Results (Phases 4-9)

End-to-end from a clean clone:

```bash
pip install -r requirements.txt

# 1. Maps (skip if environments/maps/*.json already present)
python environments/generator.py
python environments/generator.py --targets

# 2. Train (VI γ-sweep; QL 2×2×3 seeds; SARSA 4 λ × 3 seeds; 50k eps)
python experiments/run_experiments.py

# 3. Transfer (shaped reward, 5000 episodes; needs target maps + a source QL model)
python experiments/run_experiments.py transfer

# 4. Figures from saved CSVs/models
python experiments/analysis.py && python experiments/compare.py

# 5. Optional checks
pytest tests/
python gui/app.py
```

All grid parameters live in `experiment_grid` (and algorithm blocks) in
`experiments/configs/default_config.json`.

### Outputs under `results/`

| Path | Contents |
|------|----------|
| `results/raw_data/run_ledger.csv` | One row per variant attempted (success/failed) |
| `results/raw_data/{vi,q_learning,sarsa,transfer,comparison,energy}/` | Training curves, summaries, traces |
| `results/models/{vi,q_learning,sarsa,transfer}/` | Saved V/Q tables + policies (JSON) |
| `results/figures/{vi,q_learning,sarsa,comparison,transfer,misc}/` | Generated PNGs |
| `results/videos/` | Optional GUI recordings |

Notable CSVs:

- `results/raw_data/comparison/comparison_summary.csv` — Phase 8 comparison table
- `results/raw_data/comparison/comparison_sample_states.csv` — sample mismatched states (Step 54)
- `results/raw_data/transfer/transfer_summary.csv` / `transfer_training.csv` — transfer metrics

### Eligibility trace type

SARSA(lambda) uses `"trace_type": "replacing"` (see `experiments/configs/default_config.json`).
Replacing traces cap `E(s,a)` at 1.0 on revisit, preventing runaway trace magnitudes
on states visited many times within a single episode (loops near penalty cells,
repeated wall-bump retries near the energy budget). Accumulating traces
would let those revisits stack unbounded, over-weighting frequently-revisited
states in the update. The `E` dict in `agents/sarsa_lambda.py` (`SarsaLambdaAgent._apply_trace`)
is a sparse dict keyed by `(state_idx, action)`, decayed by `gamma*lambda` every
step and pruned once a trace drops below `1e-6` — this keeps updates fast even
though the full state-action space has tens of thousands of entries
(~39.5k states at `max_energy=60`).

### Trace-run seed note

`experiment_grid.*.trace_run` selects which *variant* gets a detailed step dump
(reward_mode / schedule / λ / episode_index). The training seed for that dump is
deliberately `seeds[0]` from `derive_seeds(...)`, not a separate config field —
so the traced episode always comes from a seed that was actually trained.

### Sanity-check a single episode with the logger

```bash
python -m experiments.verify_logger
```

This forces one scenario per required event type (move, wall_hit, penalty_entry,
key_pickup, door_attempt, door_open, goal_reached, step_cap, energy_depleted)
and asserts all 9 appear in the detailed per-step logs. Summary CSV and detail
JSON files land in `results/raw_data/verify_logger_summary.csv` and
`results/raw_data/verify_logger_details/`.

### Energy budget selection (`max_energy`)

The custom fuel feature is only meaningful if the energy budget is a *real* constraint:
big enough that the maze stays solvable, small enough that fuel still decides episodes.
Probed with `experiments/energy_sweep.py` (random-policy read + Value Iteration to
convergence for the optimality signal):

```bash
python experiments/energy_sweep.py --budgets 30 50 60 80 100 150 200 --episodes 50
```

Shortest feasible path start→key→goal = **30** (BFS-valid at every budget ≥ 30).

**The random-policy column cannot decide this on its own.** A uniform-random policy
dies of `energy_depleted` in 100% of episodes at *every* budget from 30 through 200
(goal rate 0.00, one lucky episode at 200), and `random_mean_steps` simply mirrors the
budget itself (30.0, 50.0, 60.0 …). A random walk never gets close to solving an
18×18 key→door→goal maze at any of these values, so it is not the right lens for
finding the constraint boundary. The real signal is **VI success rate**: VI computes
the *optimal* policy, so it shows where the budget becomes too tight to solve at all
(30), where it is borderline (50), and where it is tight-but-reliable (60).

| `max_energy` | n_states | random energy-depleted | random goal | VI success | VI mean return | VI mean steps | VI iters | VI runtime |
|--------------|----------|------------------------|-------------|------------|----------------|---------------|----------|------------|
| 30           | 20,088   | 1.00                   | 0.00        | **0.00**   | 15.92          | 30.0          | 31       | 16.7 s     |
| 50           | 33,048   | 1.00                   | 0.00        | 0.98       | 199.75         | 38.69         | 51       | 46.6 s     |
| **60**       | 39,528   | 1.00                   | 0.00        | **1.00**   | 203.65         | 38.75         | 61       | 68.5 s     |
| 80           | 52,488   | 1.00                   | 0.00        | 1.00       | 203.65         | 38.75         | 81       | 121.2 s    |
| 100          | 65,448   | 1.00                   | 0.00        | 1.00       | 203.65         | 38.75         | 92       | 142.8 s    |
| 150          | 97,848   | 1.00                   | 0.00        | 1.00       | 203.65         | 38.75         | 92       | 195.8 s    |
| 200          | 130,248  | 1.00                   | 0.00        | 1.00       | 203.65         | 38.75         | 92       | 256.9 s    |

**Decision rule:** smallest budget where VI still *reliably* succeeds
(`vi_eval_success_rate` ≈ 1.0).

- **30** → VI success **0.00**: the budget equals the BFS path length, leaving zero
  slack for the 0.8/0.1/0.1 stochastic transitions, so even the optimal policy cannot
  finish. Infeasible.
- **50** → VI success **0.98**: 1 failure in 50 eval episodes. Borderline — mostly
  works but not reliable, so it fails the decision rule.
- **60** → VI success **1.00** with the same optimal return/steps as every larger
  budget (203.65 / 38.75). **This is the smallest budget with a fully reliable optimal
  policy, i.e. the tight-but-solvable boundary.**

**Decision: `max_energy = 60`** (set in `experiments/configs/default_config.json`).

- It is a genuine constraint: random policy dies of fuel in 100% of episodes, and the
  budget (60) is only ~1.5× the ~39 energy units the optimal policy actually needs —
  fuel still binds.
- It is solvable: VI converges (61 iters) and its greedy policy wins 100% of evals.
- Budgets 80–200 add no optimality (identical 1.00 / 203.65 / 38.75) but strictly
  grow the MDP (52,488 → 130,248 states) and VI runtime (121 s → 257 s). VI iteration
  count even plateaus at 92 once the budget stops binding, so beyond ~100 the extra
  state space buys nothing.
- Note that even at 200 the *random* policy only reaches the goal on 1/200 episodes,
  so the earlier max_energy=200 "failure" was about the budget ceasing to bind an
  *optimal* agent — energy never stops being fatal for a no-skill policy.

## Project Structure

```
RL_FinalProject_40406934/
├── environments/       # Maze env, generator, saved maps
├── agents/             # Value Iteration, Q-Learning, SARSA(λ)
├── transfer/           # Transfer learning scenarios
├── gui/                # customtkinter app (`gui/app.py`)
├── experiments/        # Experiment runners, analysis, configs
├── results/
│   ├── raw_data/       # CSVs: vi, q_learning, sarsa, transfer, comparison, energy
│   ├── models/         # Saved Q-tables / V-tables (vi, q_learning, sarsa, transfer)
│   ├── figures/        # Plots (vi, q_learning, sarsa, comparison, transfer, misc)
│   └── videos/         # GUI recordings
├── tests/              # pytest unit tests
├── main.py             # Single-algo CLI dispatcher
├── requirements.txt
└── README.md
```

## Results Summary

<!-- TODO: fill in key numbers after experiments -->

## AI Usage Disclosure

| Section | AI suggestion received | Change made by student | Reason for change |
|---------|----------------------|----------------------|-------------------|
| <!-- TODO: add ≥2 real examples after implementation --> | | | |

## References

<!-- TODO: cite textbooks, papers, and any adapted code snippets -->
