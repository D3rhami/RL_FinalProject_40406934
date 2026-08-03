import argparse
import json
import time
from pathlib import Path

from environments.maze import MazeEnv, load_config
from experiments.logger import EpisodeLogger


def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def run_value_iteration(env, cfg, args):
    from agents.value_iteration import ValueIteration
    gamma = args.gamma or cfg['value_iteration']['gamma']
    theta = cfg['value_iteration']['theta']
    vi = ValueIteration(env, gamma=gamma, theta=theta, reward_mode=args.reward_mode)
    t0 = time.time()
    V, n_iter, converged, history = vi.run(max_iterations=args.max_iterations)
    policy = vi.extract_policy(V)
    states = env.all_states()
    runtime = time.time() - t0
    print(f"VI gamma={gamma} converged={converged} iterations={n_iter} "
          f"runtime={runtime:.2f}s")

    model_path = f'results/models/vi/vi_{args.reward_mode}_gamma{gamma}.json'
    _save_json(model_path, {
        'algorithm': 'value_iteration', 'gamma': gamma,
        'theta': theta, 'reward_mode': args.reward_mode,
        'iterations': n_iter, 'runtime_seconds': runtime,
        'converged': converged, 'deltas': history,
        'states': [list(s) for s in states],
        'V': V.tolist(),
        'policy': [policy.get(s, -1) for s in states],
    })
    print(f"saved {model_path}")
    return vi


def run_q_learning(env, cfg, args):
    from agents.q_learning import QLearningAgent
    qcfg = cfg['q_learning']
    decay_type = args.schedule or 'linear'
    decay_param = (qcfg['linear_decay_episodes'] if decay_type == 'linear'
                   else qcfg['exponential_decay_rate'])
    seed = args.seed if args.seed is not None else cfg['base_seed']
    agent = QLearningAgent(
        env, alpha=qcfg['alpha'], gamma=qcfg['gamma'],
        epsilon_start=qcfg['epsilon_start'], epsilon_min=qcfg['epsilon_min'],
        decay_type=decay_type, decay_param=decay_param,
        num_episodes=args.num_episodes or qcfg['num_episodes'],
        reward_mode=args.reward_mode, seed=seed,
    )
    out_dir = Path('results/raw_data/q_learning')
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(str(out_dir / f'q_learning_{decay_type}_cli_summary.csv'))
    agent.train(logger=logger)
    logger.flush_summary()

    states = env.all_states()
    state_indices = [env.encode_state(s) for s in states]
    model_path = (f'results/models/q_learning/'
                  f'q_learning_{args.reward_mode}_{decay_type}_seed{seed}.json')
    _save_json(model_path, {
        'algorithm': 'q_learning', 'alpha': qcfg['alpha'],
        'gamma': qcfg['gamma'], 'reward_mode': args.reward_mode,
        'schedule': decay_type, 'seed': seed,
        'episodes': agent.num_episodes,
        'epsilon_start': qcfg['epsilon_start'],
        'epsilon_min': qcfg['epsilon_min'],
        'decay_param': decay_param,
        'states': [list(s) for s in states],
        'Q': agent.Q[state_indices].tolist(),
        'visits': agent.visits[state_indices].tolist(),
    })
    print(f"Q-Learning ({decay_type}) done: {len(logger.summary_rows)} episodes logged")
    print(f"saved {model_path}")
    return agent


def run_sarsa_lambda(env, cfg, args):
    from agents.sarsa_lambda import SarsaLambdaAgent
    scfg = cfg['sarsa_lambda']
    lam = args.lam if args.lam is not None else scfg['lambda_sweep'][0]
    seed = args.seed if args.seed is not None else cfg['base_seed']
    agent = SarsaLambdaAgent(
        env, alpha=scfg['alpha'], gamma=scfg['gamma'], lam=lam,
        trace_type=scfg['trace_type'],
        epsilon_start=scfg['epsilon_start'], epsilon_min=scfg['epsilon_min'],
        decay_type='exponential', decay_param=scfg['exponential_decay_rate'],
        num_episodes=args.num_episodes or scfg['num_episodes'],
        reward_mode=args.reward_mode, seed=seed,
    )
    out_dir = Path('results/raw_data/sarsa')
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = EpisodeLogger(str(out_dir / f'sarsa_lambda_{lam}_cli_summary.csv'))
    agent.train(logger=logger)
    logger.flush_summary()

    states = env.all_states()
    state_indices = [env.encode_state(s) for s in states]
    model_path = (f'results/models/sarsa/'
                  f'sarsa_{args.reward_mode}_lambda{lam}_seed{seed}.json')
    _save_json(model_path, {
        'algorithm': 'sarsa_lambda', 'alpha': scfg['alpha'],
        'gamma': scfg['gamma'], 'lambda': lam,
        'trace_type': scfg['trace_type'],
        'reward_mode': args.reward_mode, 'seed': seed,
        'episodes': agent.num_episodes,
        'epsilon_start': scfg['epsilon_start'],
        'epsilon_min': scfg['epsilon_min'],
        'decay_param': scfg['exponential_decay_rate'],
        'states': [list(s) for s in states],
        'Q': agent.Q[state_indices].tolist(),
        'visits': agent.visits[state_indices].tolist(),
    })
    print(f"SARSA(lambda={lam}) done: {len(logger.summary_rows)} episodes logged")
    print(f"saved {model_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(description='RL Final Project dispatcher')
    parser.add_argument('--algo', choices=['value_iteration', 'q_learning', 'sarsa_lambda'],
                         default=None, help='Algorithm to run; omit to launch the GUI')
    parser.add_argument('--reward-mode', dest='reward_mode', choices=['sparse', 'shaped'],
                         default='sparse')
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--max-iterations', dest='max_iterations', type=int, default=1000)
    parser.add_argument('--schedule', choices=['linear', 'exponential'], default=None)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--num-episodes', dest='num_episodes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.algo is None:
        print("No --algo given. GUI entry point not yet implemented (Phase 10).")
        return

    cfg = load_config()
    env = MazeEnv(cfg['maze_file'], config=cfg, reward_mode=args.reward_mode, seed=args.seed)

    if args.algo == 'value_iteration':
        run_value_iteration(env, cfg, args)
    elif args.algo == 'q_learning':
        run_q_learning(env, cfg, args)
    elif args.algo == 'sarsa_lambda':
        run_sarsa_lambda(env, cfg, args)


if __name__ == '__main__':
    main()
