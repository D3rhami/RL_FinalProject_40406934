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

A stochastic 18×18 maze MDP (`environments/maze.py`) with a start → key → closed
door → goal structure, ≥15% walls, ≥5 penalty cells, and a **limited-energy**
extra mechanic (energy is part of the state and a hard terminal condition).
Three algorithms are implemented from scratch and compared on the identical map
and reward definition: **Value Iteration** (model-based), **Q-Learning**
(model-free, off-policy), and **SARSA(λ)** (model-free, on-policy with
eligibility traces). A limited transfer-learning study reuses a trained
Q-Learning table on two derived target maps. A `customtkinter` GUI drives the
same environment/agents live. The full analytical write-up (MDP definition,
per-algorithm results, comparison, transfer, Q1–Q6) is in `report/draft.html`;
this README covers install/run/reproduce and points out a few
implementation subtleties worth knowing before you read the report.

## Environment & MDP

- **State** `s = (r, c, has_key, energy)` — position, whether the key has been
  collected, and remaining energy. Position alone is not Markov here: whether
  the door blocks the agent depends on `has_key`, and the future horizon
  depends on `energy`, so both are folded into the state (`environments/maze.py`,
  `MazeEnv.encode_state` / `all_states`).
- **Actions** `{UP, DOWN, LEFT, RIGHT}`.
- **Transition**: intended action with probability `p_intended=0.8`, each
  perpendicular direction with `p_perp=0.1` each. A wall/boundary collision
  leaves the agent in place and applies `wall_hit` reward. Attempting the door
  without the key also leaves the agent in place (`door_attempt` penalty).
  Energy decreases by 1 every step regardless of outcome.
- **Rewards**: two versions in `experiments/configs/default_config.json` —
  `sparse` (`rewards` block: step/wall/penalty/key/door/goal only) and
  `shaped` (`shaping` block: potential-based distance shaping toward the
  key/goal, a capped safe-passage bonus near penalty cells, and a wasted-move
  penalty). Both are exercised through the pipeline; see report §2 for the
  numeric justification and the sparse-vs-shaped comparison.
- **Termination**: goal reached with key, energy hits 0, or steps exceed the
  configured cap (`3 × walkable_cells`, `env.max_steps_multiplier`).

### Why the on-screen policy changes right after key pickup (not a bug)

The door in `source_maze.json` is not a physical bottleneck separate from the
key — reaching the goal cell only counts as a win when `has_key == 1`
(`MazeEnv.step`), so the key is a *reward-gated precondition*, not a second
maze region to route around. Q/V are stored per full state `(r, c, has_key,
energy)`, so the optimal action at the *same cell* can legitimately differ
before vs. after key pickup (e.g. "route toward the key" vs. "route toward the
goal"). What the GUI's policy overlay shows after key pickup is simply the
`has_key=1` slice of the same table — not a re-learned or corrupted policy.
Cells that look blank in the overlay are states that were rarely or never
visited during training (their Q-row never moved off zero), not an error in
the slicing logic (`gui/app.py`, `_load_policy_grid` / `_slice_has_key_energy`).

## Algorithms

### Value Iteration (`agents/value_iteration.py`)

From-scratch Bellman backup `V(s) ← max_a Σ P(s'|s,a)[R + γV(s')]` with a
max-ΔV convergence threshold (`theta`), run at 3 `γ` values
(`experiments/configs/default_config.json → value_iteration.gamma_sweep`).
Outputs: `results/models/vi/vi_*.json` (V, policy, iteration count, runtime),
`results/figures/vi/vi_value_policy.png` (heatmap + greedy arrows),
`results/figures/vi/vi_gamma_sensitivity.png`, `vi_convergence.png`. VI's
greedy policy is the reference used for the cross-algorithm agreement metric.

### Q-Learning (`agents/q_learning.py`)

Off-policy `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]`, ε-greedy
behaviour with **two** ε-decay schedules (linear, exponential), trained for
both `sparse` and `shaped` reward (2×2 grid × 3 seeds, 50 000 episodes each).
Per-episode reward/steps/success/wall-hits/penalty-entries are logged to
`results/raw_data/q_learning/q_learning_training.csv`; one real Q-update is
hand-reconstructed from `q_update_trace.csv` in the report (§4.2).

### SARSA(λ) (`agents/sarsa_lambda.py`)

On-policy `δ = r + γQ(s',a') − Q(s,a)`, **replacing** eligibility traces
(`E(s,a) ← γλE(s,a) + 1{s=s_t,a=a_t}`, capped at 1 on revisit — see
"Eligibility trace type" below), swept over `λ ∈ {0, 0.3, 0.7, 0.9}`.

**Reward-mode fix (this audit).** SARSA's experiment grid originally only
ever trained with `reward_mode=sparse`
(`experiment_grid.sarsa_lambda.reward_mode`, singular field), while
Q-Learning's grid always trained both `sparse` **and** `shaped`
(`experiment_grid.q_learning.reward_modes`, a list). That asymmetry — not
`max_energy=60` — is what made SARSA's success rate look broken (λ=0.9 hit
**0.0** eval success on all 3 seeds; even λ=0.3/0.7 only reached ~1–10%):
SARSA never got the same shaped-reward benefit that took Q-Learning from
~11% to ~43% eval success. The config now has
`experiment_grid.sarsa_lambda.reward_modes: ["sparse", "shaped"]` (plus a
`reference_reward_mode: "sparse"` used only for the Phase-8 cross-algorithm
comparison, see below), `run_experiments.py` trains both, and
`experiments/analysis.py` adds a `sarsa_rewardmode_comparison.png` figure
mirroring Q-Learning's. `max_energy=60` itself is independently justified —
see "Energy budget selection" below — and was left unchanged.

### Cross-Algorithm Comparison (`experiments/compare.py`)

VI, Q-Learning, and SARSA(λ) are compared on the identical map **and identical
reward mode**, as the assignment requires. This audit fixed a real fairness
bug: `_pick_best_ql_variant`/`_pick_best_sarsa_variant` previously picked
whichever `(reward_mode, schedule/λ)` had the single highest
`eval_success_rate` — which, once shaped runs existed, silently selected
shaped Q-Learning against sparse VI/SARSA. Both functions now restrict the
search to each algorithm's `experiment_grid.*.reference_reward_mode`
(`sparse` for all three) before picking the best schedule/λ, so
`results/raw_data/comparison/comparison_summary.csv` compares like-for-like.
Runtime, samples-to-90%, cross-seed/within-run stability, path quality
(eval mean steps), and % greedy-action agreement with VI (overall,
visited-only, near-penalty) are all reported; `policy_grid.png` /
`agreement_grid.png` visualize agreement, and
`comparison_sample_states.csv` lists concrete mismatched states analyzed in
the report (§6.2).

## Transfer Learning

Q-Learning only, per the assignment. A source Q-table is trained on
`source_maze.json`, then reused on two BFS-validated target maps generated by
`environments/generator.py --targets`:

- **similar**: `similar_obstacle_change_pct=0.175` (15–20% obstacles moved),
  start/key/goal unchanged.
- **different**: `different_obstacle_change_pct=0.40` (≥35% obstacles
  changed), key/goal relocated, extra penalty cells added.

`transfer/transfer_learning.py` runs the full experiment matrix
(`_scenario_specs`): **scratch** (zero-init baseline), **full** (copy the
entire source Q-table), **scaled** at β ∈ {0.25, 0.50, 0.75}
(`Q_T⁽⁰⁾ = βQ_S`), and **selective** (copy only states whose local wall
neighborhood is unchanged, via `unchanged_state_indices`) — 6 scenario
variants × 2 targets × 3 seeds = 36 runs, plus a dedicated negative-transfer
case study (`run_negative_transfer_case`). Initial (jumpstart), learning-speed
(episodes-to-90%), and final (eval) performance are reported separately per
scenario/target in `results/raw_data/transfer/transfer_summary.csv` and
plotted in `results/figures/transfer/transfer_{similar,different}_comparison.png`.
`transfer_maze_{similar,different}.png` shows the source maze and the target
maze side by side with each agent's real greedy rollout, so the structural
differences between the two maps and how the transferred agent behaves on the
new layout are visible together. `transfer_q_diff.png` shows the spatial
|ΔQ| before/after fine-tuning, and `negative_transfer_recovery.png` /
`negative_transfer_case.json` document one concrete case where transferred
knowledge initially pointed the wrong way and how continued training on the
target corrected it. See report §7 for the full numeric comparison and Q6.

## GUI

`customtkinter` app (`gui/app.py` + `gui/renderer.py`) with step-by-step
animated rollout, live info panel (episode/step/reward/ε/key/energy/recent
success rate), algorithm selector (VI/Q-Learning/SARSA(λ)), train/eval mode
toggle with an episode-count input for training, start/stop/continue/reset/
rerun controls, animation-speed slider, and a policy-overlay toggle that shows
per-cell greedy arrows for the agent's *current* `(has_key, energy)` slice.

- **Target-map selector.** Picking a transfer target map is a secondary,
  advanced option (most of the assignment's required controls concern the
  *source* map/algorithm), so it's deliberately a small, muted dropdown
  tucked in the bottom-right corner of the maze view (`env_corner` /
  `env_menu` in `gui/app.py`) rather than a full-width control competing with
  Algorithm/Mode in the primary CONTROLS row.
- **Transfer scenario picker.** When a target map (`similar`/`different`) is
  selected, a `Transfer scenario` dropdown appears in CONTROLS
  (`scenario_menu`) letting you switch between all 6 trained scenarios
  (`scratch`, `full`, `scaled_0.25/0.5/0.75`, `selective`) — previously the
  GUI only ever loaded the `full` scenario model.

## Installation

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
| `env.max_energy` | `500` |

Training episode counts in the same config:

- Q-Learning / SARSA(λ): **20,000** episodes (`q_learning.num_episodes`, `sarsa_lambda.num_episodes`)
- Transfer: **shaped** reward, **5,000** episodes (`transfer.reward_mode`, `transfer.num_episodes`)

### 20k vs 50k episodes

`q_learning.num_episodes` / `sarsa_lambda.num_episodes` were lowered from
50,000 to **20,000** to keep the full rerun (12 QL variants + 24 SARSA
variants × 3 seeds each, on a `max_energy=500` state space of 324,648
states) tractable under a tight schedule. This is a safe reduction, not a
shortcut: both ε-decay schedules already reach `epsilon_min` well before
20k episodes (linear by episode 5,000, exponential by ~episode 920), so the
dropped 30k episodes were almost all greedy-refinement episodes, not
exploration. The energy-budget learnability probe above
(`experiments/energy_sweep.py`) independently trained a single-seed,
**20,000-episode**, shaped-reward agent at this exact `max_energy=500` and
got QL eval success **0.86** / SARSA(λ=0.3) eval success **0.71** — direct
evidence that 20k episodes already learns a strong policy at this energy
budget. Sparse-reward variants are expected to lag shaped ones more at 20k
than they would at 50k (weaker learning signal, needs more exploration to
find the same policy); that gap is itself part of the sparse-vs-shaped
analysis, not a bug. Both `q_learning.num_episodes` and
`sarsa_lambda.num_episodes` remain single config values applied uniformly
across every variant/seed, so all comparisons stay fair and reproducible.

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

QL/SARSA train for **20,000** episodes; transfer uses **shaped** reward and
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

# 2. Train (VI γ-sweep; QL 2×2×3 seeds; SARSA 4 λ × 3 seeds; 20k eps)
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

### Energy budget selection (`max_energy`) — revised after a learnability audit

**This project went through two rounds of energy selection.** The first round
(below, "VI-only view") only checked whether the budget was *solvable at all*
and picked the smallest such budget (60). The second round (this section)
adds a check the first round was missing: whether **model-free learners
(Q-Learning/SARSA) can actually make progress** at that budget within a
realistic episode count — and found that 60 was a poor choice by that
criterion, even though VI "solves" it trivially.

**Why VI success alone is the wrong criterion.** VI gets the exact transition
model and computes the optimum directly — it never has to *survive* an energy
budget while still exploring randomly. Q-Learning/SARSA start ε-greedy and
have to accumulate enough successful (or partially successful) episodes to
learn anything; if the budget is so tight that almost every early, undirected
rollout dies of `energy_depleted` before it ever reaches the key, the reward
signal needed to learn is starved. `experiments/energy_sweep.py --with-learners`
makes this concrete: it trains a short, single-seed, shaped-reward Q-Learning
and SARSA(λ=0.3) run at each candidate budget and reports **greedy eval
success**, not just whether VI can solve it.

```bash
# VI-only view (fast, no learners) — establishes solvability floor
python experiments/energy_sweep.py --budgets 30 50 60 80 100 150 200 --episodes 50

# Learnability view (adds a short QL + SARSA probe per budget)
python experiments/energy_sweep.py --budgets 60 100 150 200 300 400 500 \
    --skip-vi --with-learners --learn-episodes 20000 --learn-eval-episodes 100
```

**VI-only view** (`n_states`, random-policy behaviour, VI eval success) —
shortest feasible path start→key→goal = **30**, so budgets ≥ 30 are BFS-valid:

| `max_energy` | n_states | random goal rate | VI success | VI mean steps |
|--------------|----------|-------------------|------------|----------------|
| 30           | 20,088   | 0.00              | **0.00**   | 30.0           |
| 50           | 33,048   | 0.00              | 0.98       | 38.69          |
| 60           | 39,528   | 0.00              | **1.00**   | 38.75          |
| 80–200       | 52k–130k | 0.00–0.01         | 1.00       | 38.75          |

By this view alone, 60 looks like "the smallest fully-reliable budget" — but
VI's mean steps (38.75) barely change across the whole range, which is itself
the tell: **a competent policy's actual energy usage doesn't depend on the
budget**, so the "slack" (`max_energy − 38.75`) is the real free parameter,
and the VI-only view says nothing about whether a *sample-based* learner can
find that policy in the first place.

**Learnability view** — single-seed, 20,000-episode shaped-reward probe,
100 greedy eval episodes (`results/raw_data/energy/energy_sweep.csv`):

| `max_energy` | n_states | slack over VI-optimal (38.75) | QL probe eval success | SARSA(0.3) probe eval success |
|--------------|----------|-------------------------------|-----------------------|-------------------------------|
| 60           | 39,528   | 1.5×                          | **0.18** (6k-ep probe)| **0.15** (6k-ep probe)        |
| 100          | 65,448   | 2.6×                          | 0.60                  | 0.66                          |
| 130          | 84,888   | 3.4×                          | 0.69                  | 0.62                          |
| 150          | 97,848   | 3.9×                          | 0.70                  | 0.50                          |
| 200          | 130,248  | 5.2×                          | 0.78                  | 0.51                          |
| 300          | 195,048  | 7.7×                          | 0.71                  | 0.52                          |
| 400          | 259,848  | 10.3×                         | 0.73                  | 0.69                          |
| **500**      | 324,648  | **12.9×**                     | **0.86**              | **0.71**                      |

(60's row used the earlier 6,000-episode probe; 100–500 used the 20,000-episode
probe, which now matches the real training budget (see "20k vs 50k episodes"
note below) — these are still relative learnability signals from a
single-seed run, not the final multi-seed numbers.)

**Trade-off, stated honestly.** Raising the budget helps model-free learners
in two ways: fewer early rollouts die before finding any reward signal, and
the eventual optimal policy has more room for slip-induced detours without
running out of fuel. But it also grows the state space linearly (39.5k → 325k
states at 500), which for a *fixed* training budget dilutes visits per state —
and, more importantly, at very large slack (12.9× the optimal path) the
energy constraint stops being something a trained policy ever really feels,
which is in tension with the assignment's requirement that the custom
mechanic have "a real effect, not just visual." The learnability numbers
above don't show a clean monotonic story (150→200→300 wobbles for SARSA) —
consistent with this being a single-seed, short-probe signal, not a precise
optimum.

**Decision: `max_energy = 500`.** Chosen because it gave the strongest
model-free learnability signal in the probe (QL 0.86 / SARSA 0.71, both the
best of the candidates tested) and the project prioritizes giving Q-Learning,
SARSA(λ), and the transfer-learning scenarios a realistic chance to reach
non-trivial success rates within the actual training budget — over keeping
the energy constraint as tight as mathematically possible. The full
multi-seed retraining under this budget (VI + QL + SARSA + transfer;
`experiments/run_experiments.py --fresh`) is the real confirmation of this
choice; the probe above is a fast (~15 minutes total) way to search the space
before committing to the full rerun. `max_energy` and its sweep
candidates live in `experiments/configs/default_config.json` → `env`.

### Cleanliness / reproducibility note

Every figure under `results/figures/` and every model/CSV under
`results/models/` and `results/raw_data/` is produced directly by
`experiments/run_experiments.py`, `experiments/analysis.py`, and
`experiments/compare.py` from the config above — a clean clone + the
"Reproducing Results" steps regenerates all of it with no manual edits.
Stray diagnostic/debug artifacts found during this audit (ad-hoc energy
probes, superseded comparison figures from before the reward-mode fairness
fix, an old `gui_checks/` dump) were moved to `tmp/old_results_backup/`
rather than left mixed into `results/`, so `results/` only ever contains
outputs a grader can regenerate from the committed scripts/config.

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

Full numbers, tables, and per-chart analysis live in `report/draft.html`
(every claim there cites a specific CSV/figure under `results/`). Headline
findings:

- **VI** solves the map perfectly at every tested γ (eval success **1.0**,
  ~38.75 steps); γ mostly rescales `V(start)`, not the greedy policy.
- **Q-Learning**: shaping is a large win — shaped/linear reaches the best
  *final* greedy policy, while shaped/exponential finds first success soonest
  but with a weaker final policy; sparse 20k episodes is not enough to match
  VI. See `results/figures/q_learning/`.
- **SARSA(λ)**: replacing traces avoid runaway eligibility under looping
  exploration; λ trades off eval quality vs. training speed differently
  (see report §5.1 and the "Reward-mode fix" note above for why the original
  sparse-only sweep looked broken). See `results/figures/sarsa/`.
- **Comparison**: VI is the reference optimum; both model-free methods trail
  it on policy agreement and path efficiency by a margin quantified in
  `comparison_summary.csv` and visualized in `results/figures/comparison/`.
- **Transfer**: on the *similar* target, transfer jumpstarts return and can
  match/beat training from scratch; on the *different* target (relocated
  key/goal, extra penalties), full transfer is actively harmful and scratch
  wins — see `results/figures/transfer/` and the negative-transfer case study.

## AI Usage Disclosure

AI assistants (Cursor agents) were used throughout for implementation,
debugging, plotting, and drafting experiment scripts/docs. All numbers cited
in the report and here come from CSVs/models actually produced by local runs
under `results/`, not hand-typed. The full disclosure table with ≥2 flawed
AI-suggestion examples (energy-budget myth, max-energy-only policy slicing,
a reward-mode mismatch mis-diagnosed as an energy problem, RdYlGn
colormap mash-up, generic "transfer was good" prose) is in `report/draft.html`
§9 — reproduced in short form here:

| Use case | Flawed/incomplete AI suggestion | Student correction | Why |
|----------|----------------------------------|---------------------|-----|
| SARSA stuck near 0% success | "Just double `max_energy` (60→100)" | Diagnosed the real cause instead: `sarsa_lambda`'s experiment grid only ever trained `reward_mode=sparse` while `q_learning`'s grid trained both `sparse` and `shaped`; added the missing shaped sweep and kept `max_energy=60` | Doubling energy doesn't explain λ=0.9's total (0%) failure; the asymmetric reward-mode grid does, and is fixable without re-justifying the energy budget |
| Cross-algorithm "best" model selection | Pick whichever `(reward_mode, schedule/λ)` has the single highest `eval_success_rate` | Restricted selection to each algorithm's `reference_reward_mode` (`sparse`) before ranking, in both `_pick_best_ql_variant` and `_pick_best_sarsa_variant` | The unrestricted version silently compared shaped Q-Learning against sparse VI/SARSA, violating the assignment's "identical reward definition" requirement for Phase 8 |
| Policy overlay showing one arrow | Read the greedy policy at `energy = max_energy` only | Slice by the agent's *current* `(has_key, energy)`, falling back to the most-visited energy per cell for saved models | After the first step, energy is never `max_energy` again, so the old slice showed an almost-empty policy |
| GUI entry point | Left `main.py` printing "GUI not implemented" | Wired `main.py` (no `--algo`) to launch `gui.app.App` | The GUI was fully implemented; the CLI dispatcher just never called it |

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press. — Bellman backups (VI), Q-Learning,
  SARSA(λ) with eligibility traces, ε-greedy exploration.
- Course lecture notes / assignment specification for this project
  (`tmp/final_project_text.md`) — state/action/reward design constraints,
  transfer-learning scenario definitions, and required deliverables.
- Third-party libraries only (no RL libraries): NumPy, Pandas, Matplotlib,
  `customtkinter`, `pygame` (optional), `pytest` — see `requirements.txt`.
  No code from Stable-Baselines/RLlib or similar is used anywhere in
  `agents/`.
