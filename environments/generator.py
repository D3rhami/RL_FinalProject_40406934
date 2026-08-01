"""
generator.py
------------
Seeded procedural maze generator.

Student ID : 40406934
base_seed  : int('40406934'[-2]) = 3
maze_size  : 15 + (3 % 4)       = 18  →  18×18 grid

Responsibilities
----------------
- Generate walls (≥15 % of cells) using the fixed seed
- Place ≥5 penalty cells, start, key, door, goal, and energy pickups
- Guarantee no placement overlaps
- Save the final map to environments/maps/maze_40406934.json
- Run BFS to validate start→key and key→goal reachability
- Repair/regenerate until a valid map is produced
"""

# TODO: implement generate_maze(), validate_map(), save_map()
