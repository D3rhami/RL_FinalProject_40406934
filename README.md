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

## Reproducing Results

<!-- TODO: fill in exact commands + config pointers for every figure once
     Phase 5-7's full runs (VI gamma sweep, QL both schedules, SARSA(lambda)
     sweep) have been executed locally via the commands above. -->

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
