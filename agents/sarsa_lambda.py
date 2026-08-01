"""
sarsa_lambda.py
---------------
On-policy SARSA(λ) with eligibility traces.

Trace type : replacing  (justified in report)

λ sweep    : 0, 0.3, 0.7, 0.9

Per-episode logging
-------------------
total_reward, steps, success, wall_hits, penalty_entries, epsilon

Outputs
-------
- Q-table per λ value (numpy .npy)
- Episode log per λ value (CSV)
- δ and E trace for one short episode (CSV)
"""

# TODO: implement SarsaLambda class
