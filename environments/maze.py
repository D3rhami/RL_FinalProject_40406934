"""
maze.py
-------
Defines cell/tile types and the MazEnv class (OpenAI-Gym-style interface).

Cell types
----------
EMPTY   : normal walkable cell
WALL    : impassable obstacle
PENALTY : walkable but incurs extra negative reward
START   : agent starting position
KEY     : collectible that unlocks the door
DOOR    : blocked until key is collected
GOAL    : terminal success state
ENERGY  : energy-pickup cell (chosen extra feature – restores energy units)

State representation
--------------------
(x, y, has_key, energy)  →  preserves Markov property:
  • has_key tracks whether the door can be opened
  • energy tracks remaining steps before forced termination
  Both variables affect future transitions, so history is not needed.
"""

# TODO: implement MazeEnv
