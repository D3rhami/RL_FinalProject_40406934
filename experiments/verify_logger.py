import numpy as np

from environments.maze import MazeEnv, load_config, CellType
from experiments.logger import EpisodeLogger

MAP_PATH = 'environments/maps/source_maze.json'

NEIGHBOR_OFFSETS = ((-1, 0, 1), (1, 0, 0), (0, -1, 3), (0, 1, 2))


def find_neighbor(env, target, avoid=(CellType.WALL,)):
    tr, tc = target
    for dr, dc, a in NEIGHBOR_OFFSETS:
        nr, nc = tr + dr, tc + dc
        if 0 <= nr < env.maze_size and 0 <= nc < env.maze_size:
            if int(env.grid[nr, nc]) not in avoid:
                return (nr, nc), a
    return None, None


def force_step(env, state, action, want_events, max_tries=200):
    for _ in range(max_tries):
        env.state = state
        s, r, done, info = env.step(action)
        if info['event'] in want_events or info.get('termination_reason') in want_events:
            return s, r, done, info
    return s, r, done, info


def run_scenario(env, logger, ep_idx, state, action, want_events, epsilon=None, lam=None):
    logger.start_episode(ep_idx, keep_detail=True, epsilon=epsilon, lam=lam)
    s, r, done, info = force_step(env, state, action, want_events)
    logger.log_step(state, action, r, info, done=done)
    logger.end_episode()
    return info


def main():
    cfg = load_config()
    env = MazeEnv(MAP_PATH, config=cfg, seed=0)
    logger = EpisodeLogger(
        'results/raw_data/verify_logger_summary.csv',
        detail_dir='results/raw_data/verify_logger_details',
    )

    events_seen = set()

    rng = np.random.default_rng(1)
    logger.start_episode(0, keep_detail=True)
    env.reset()
    state = env.state
    done = False
    while not done:
        a = int(rng.integers(0, 4))
        s, r, done, info = env.step(a)
        logger.log_step(state, a, r, info, done=done)
        events_seen.add(info['event'])
        if info.get('termination_reason'):
            events_seen.add(info['termination_reason'])
        state = s
    logger.end_episode()
    print(f"episode 0 (random rollout, {env.steps} steps): {sorted(events_seen)}")

    kr, kc = env.key_pos
    neighbor, action = find_neighbor(env, (kr, kc))
    state = (neighbor[0], neighbor[1], 0, env.MAX_ENERGY)
    info = run_scenario(env, logger, 1, state, action, {'key_pickup'})
    events_seen.add(info['event'])
    print(f"episode 1 (key_pickup):    {info['event']}")

    dr_pos = env.door_pos
    neighbor, action = find_neighbor(env, dr_pos)
    state = (neighbor[0], neighbor[1], 0, env.MAX_ENERGY)
    info = run_scenario(env, logger, 2, state, action, {'door_attempt'})
    events_seen.add(info['event'])
    print(f"episode 2 (door_attempt):  {info['event']}")

    state = (neighbor[0], neighbor[1], 1, env.MAX_ENERGY)
    info = run_scenario(env, logger, 3, state, action, {'door_open'})
    events_seen.add(info['event'])
    print(f"episode 3 (door_open):     {info['event']}")

    gr, gc = env.goal_pos
    neighbor, action = find_neighbor(env, (gr, gc))
    state = (neighbor[0], neighbor[1], 1, env.MAX_ENERGY)
    info = run_scenario(env, logger, 4, state, action, {'goal_reached'})
    events_seen.add(info['event'])
    if info.get('termination_reason'):
        events_seen.add(info['termination_reason'])
    print(f"episode 4 (goal_reached):  {info['event']} term={info.get('termination_reason')}")

    pr, pc = env.penalty_cells[0]
    neighbor, action = find_neighbor(env, (pr, pc), avoid=(CellType.WALL, CellType.DOOR, CellType.PENALTY))
    state = (neighbor[0], neighbor[1], 0, env.MAX_ENERGY)
    info = run_scenario(env, logger, 5, state, action, {'penalty_entry'})
    events_seen.add(info['event'])
    print(f"episode 5 (penalty_entry): {info['event']}")

    state = (0, 0, 0, env.MAX_ENERGY)
    info = run_scenario(env, logger, 6, state, 0, {'wall_hit'})
    events_seen.add(info['event'])
    print(f"episode 6 (wall_hit):      {info['event']}")

    state = (env.start_pos[0], env.start_pos[1], 0, 1)
    info = run_scenario(env, logger, 7, state, 2, {'energy_depleted'})
    events_seen.add(info.get('termination_reason'))
    print(f"episode 7 (energy_depleted): term={info.get('termination_reason')}")

    env.max_steps = 1
    state = (env.start_pos[0], env.start_pos[1], 0, env.MAX_ENERGY)
    info = run_scenario(env, logger, 8, state, 2, {'step_cap'})
    events_seen.add(info.get('termination_reason'))
    print(f"episode 8 (step_cap):      term={info.get('termination_reason')}")

    logger.flush_summary()

    required = {'move', 'wall_hit', 'penalty_entry', 'key_pickup', 'door_attempt',
                'door_open', 'goal_reached', 'step_cap', 'energy_depleted'}
    missing = required - events_seen
    print()
    print(f"required event types : {sorted(required)}")
    print(f"events observed      : {sorted(events_seen)}")
    print(f"missing              : {sorted(missing) if missing else 'NONE'}")
    assert not missing, f"Missing event types: {missing}"
    print("PASS: all 9 event types confirmed in logs")


if __name__ == '__main__':
    main()
