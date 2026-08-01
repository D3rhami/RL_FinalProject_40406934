"""
run_experiments.py
------------------
Runs all experiments end-to-end and saves raw results.

Usage
-----
    python experiments/run_experiments.py

Experiment groups
-----------------
1. Value Iteration    – γ sweep {0.90, 0.95, 0.99}
2. Q-Learning         – linear vs exponential ε-decay
3. SARSA(λ)           – λ sweep {0, 0.3, 0.7, 0.9}
4. Cross-algorithm comparison on identical map + reward
5. Transfer learning  – all 4 scenarios × 2 target envs

All outputs go to results/raw_data/ and results/models/.
"""

# TODO: implement experiment runners
