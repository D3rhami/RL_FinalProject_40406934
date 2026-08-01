"""
transfer_learning.py
--------------------
Q-table transfer experiments (Q-Learning only).

Target environments
-------------------
- Similar  : ~15-20 % obstacles moved, start/key/goal fixed
- Different : ≥35 % obstacles changed, key/goal moved, new penalty cells

Scenarios
---------
1. From scratch (zero Q-table baseline)
2. Full transfer  : Q_T = Q_source
3. Scaled transfer: Q_T = β * Q_source  for β ∈ {0.25, 0.50, 0.75}
4. Selective transfer: only states whose local neighbourhood is unchanged

Metrics
-------
initial_performance, learning_speed, final_performance
Identify ≥1 negative-transfer case (specific state + Q-values).
"""

# TODO: implement TransferLearning class
