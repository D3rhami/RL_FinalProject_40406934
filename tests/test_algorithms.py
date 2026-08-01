"""
test_algorithms.py
------------------
Unit tests for RL algorithm correctness.

Coverage targets
----------------
- Bellman backup produces expected V for a tiny hand-computed MDP
- Q-update arithmetic matches manual calculation from a log entry
- SARSA(λ) eligibility trace update (accumulating and replacing)
- ε-decay schedules reach epsilon_min within expected episodes
"""

# TODO: implement tests using pytest
