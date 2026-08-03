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
7. [Running the Project](#running-the-project)
8. [Reproducing Results](#reproducing-results)
9. [Project Structure](#project-structure)
10. [Results Summary](#results-summary)
11. [AI Usage Disclosure](#ai-usage-disclosure)
12. [References](#references)

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

## Running the Project

```bash
# Launch the Pygame GUI (Phase 10)
python main.py

# Run one algorithm via the CLI dispatcher
python main.py --algo value_iteration --gamma 0.95 --max-iterations 1000
python main.py --algo q_learning --schedule linear --num-episodes 10000
python main.py --algo q_learning --schedule exponential --num-episodes 10000
python main.py --algo sarsa_lambda --lam 0.7 --num-episodes 10000

# Run all experiments headlessly
python experiments/run_experiments.py

# Run unit tests
pytest tests/
```

### Sanity-check a single episode with the logger

```bash
python -m experiments.verify_logger
```

This forces one scenario per required event type (move, wall_hit, penalty_entry,
key_pickup, door_attempt, door_open, goal_reached, step_cap, energy_depleted)
and asserts all 9 appear in the detailed per-step logs. Summary CSV and detail
JSON files land in `results/raw_data/verify_logger_summary.csv` and
`results/raw_data/verify_logger_details/`.

### Eligibility trace type

SARSA(lambda) uses `"trace_type": "replacing"` (see `experiments/configs/default_config.json`).
Replacing traces cap `E(s,a)` at 1.0 on revisit, preventing runaway trace magnitudes
on states visited many times within a single episode (loops near penalty cells,
repeated wall-bump retries near the tight 66-energy budget). Accumulating traces
would let those revisits stack unbounded, over-weighting frequently-revisited
states in the update. The `E` dict in `agents/sarsa_lambda.py` (`SarsaLambdaAgent._apply_trace`)
is a sparse dict keyed by `(state_idx, action)`, decayed by `gamma*lambda` every
step and pruned once a trace drops below `1e-6` — this keeps updates fast even
though the full state-action space has ~174,000 entries.

## Reproducing Results (Phases 4-8)

Full experiment grid: VI gamma-sweep (3 runs), Q-Learning (2 reward_modes x 2
schedules x 3 seeds = 12 runs), SARSA(lambda) (4 lambdas x 3 seeds = 12 runs).
All parameters live in `experiment_grid` in `experiments/configs/default_config.json`.

```bash
# Run everything (VI ~200s, QL ~100s, SARSA ~200s -- a few minutes total)
python experiments/run_experiments.py

# Or run a subset (re-run just one algorithm after a fix)
python experiments/run_experiments.py vi
python experiments/run_experiments.py q_learning
python experiments/run_experiments.py sarsa

# Generate all figures from the saved CSVs/models (Steps 28, 31, 39, 48)
python experiments/analysis.py

# Phase 8 cross-algorithm comparison (run only after the above succeeds)
python experiments/compare.py
```

Outputs:
- `results/raw_data/run_ledger.csv` -- one row per variant attempted (success/failed)
- `results/raw_data/{vi,q_learning,sarsa}/*.csv` -- training curves + summaries
- `results/models/{vi,q_learning,sarsa}/*.json` -- saved V/Q tables + policies
- `results/figures/{vi,q_learning,sarsa,comparison}/*.png` -- all required figures
- `results/raw_data/comparison/comparison_summary.csv` -- Phase 8 comparison table
- `results/raw_data/comparison/comparison_sample_states.csv` -- the 3 required sample states (Step 54)

## Project Structure

```
RL_FinalProject_40406934/
├── environments/       # Maze env, generator, saved maps
├── agents/             # Value Iteration, Q-Learning, SARSA(λ)
├── transfer/           # Transfer learning scenarios
├── gui/                # Pygame app and renderer
├── experiments/        # Experiment runners, analysis, configs
├── results/
│   ├── raw_data/       # Per-episode CSV logs
│   ├── models/         # Saved Q-tables / V-tables
│   ├── figures/        # All saved plot images
│   └── videos/         # GUI recordings
├── tests/              # pytest unit tests
├── main.py
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
