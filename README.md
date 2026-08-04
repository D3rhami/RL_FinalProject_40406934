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
# Launch the GUI (Phase 10 — not yet wired)
python main.py

# Single-algorithm CLI (saves models under results/models/ in the same schema
# as the full pipeline; useful for a quick smoke check or one-off re-run)
python main.py --algo value_iteration --gamma 0.95 --max-iterations 1000
python main.py --algo q_learning --schedule linear --num-episodes 10000
python main.py --algo q_learning --schedule exponential --num-episodes 10000
python main.py --algo sarsa_lambda --lam 0.7 --num-episodes 10000

# Full experiment grid (the real pipeline for analysis/compare)
python experiments/run_experiments.py

# Energy-budget probe (shows the custom fuel feature has a real effect)
python experiments/energy_sweep.py

# Run unit tests
pytest tests/
```

`main.py --algo …` writes a single model JSON (+ a summary CSV for QL/SARSA)
compatible with `results/models/{vi,q_learning,sarsa}/`. It does **not** produce
the full multi-seed / multi-variant CSVs that `analysis.py` and `compare.py`
expect — use `experiments/run_experiments.py` for that.

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
repeated wall-bump retries near the energy budget). Accumulating traces
would let those revisits stack unbounded, over-weighting frequently-revisited
states in the update. The `E` dict in `agents/sarsa_lambda.py` (`SarsaLambdaAgent._apply_trace`)
is a sparse dict keyed by `(state_idx, action)`, decayed by `gamma*lambda` every
step and pruned once a trace drops below `1e-6` — this keeps updates fast even
though the full state-action space has ~65,000 entries at `max_energy=100`.

### Trace-run seed note

`experiment_grid.*.trace_run` selects which *variant* gets a detailed step dump
(reward_mode / schedule / λ / episode_index). The training seed for that dump is
deliberately `seeds[0]` from `derive_seeds(...)`, not a separate config field —
so the traced episode always comes from a seed that was actually trained.

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

**Decision: `max_energy = 60`.**

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

**Caveat:** `experiments/configs/default_config.json` still sets `"max_energy": 100`.
The final full-grid runs (Phases 4–8) should be re-run at 60 for the tighter,
smaller MDP before locking in the numbers in this README.

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
