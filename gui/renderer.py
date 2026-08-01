"""
renderer.py
-----------
Pygame renderer: draws the maze grid and overlays.

Visual elements
---------------
- WALL       : dark grey fill
- EMPTY      : light fill
- PENALTY    : red tint
- START      : green marker
- KEY        : yellow icon
- DOOR       : brown (closed) / open gap (open)
- GOAL       : gold star
- AGENT      : blue circle
- ENERGY     : cyan bolt icon  (extra feature)

Overlays
--------
- Value heatmap        (V or max_a Q)
- Policy arrows        (best action per cell)
- Visitation heatmap   (visit counts)
- Policy-difference map (agree/disagree with VI)
- Q-difference map     (pre/post transfer)
"""

# TODO: implement Renderer class
