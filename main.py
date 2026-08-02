import argparse
import time

from environments.maze import MazeEnv, load_config
from experiments.logger import EpisodeLogger


def run_value_iteration(env, cfg, args):
    from agents.value_iteration import ValueIteration
    gamma = args.gamma or cfg['value_iteration']['gamma']
    theta = cfg['value_iteration']['theta']
    vi = ValueIteration(env, gamma=gamma, theta=theta, reward_mode=args.reward_mode)
    t0 = time.time()
    V, n_iter, converged, history = vi.run(max_iterations=args.max_iterations)
    print(f"VI gamma={gamma} converged={converged} iterations={n_iter} "
          f"runtime={time.time()-t0:.2f}s")
    return vi


def run_q_learning(env, cfg, args):
    from agents.q_learning import QLearningAgent
    qcfg = cfg['q_learning']
    decay_type = args.schedule or 'linear'
    decay_param = (qcfg['linear_decay_episodes'] if decay_type == 'linear'
                   else qcfg['exponential_decay_rate'])
    agent = QLearningAgent(
        env, alpha=qcfg['alpha'], gamma=qcfg['gamma'],
        epsilon_start=qcfg['epsilon_start'], epsilon_min=qcfg['epsilon_min'],
        decay_type=decay_type, decay_param=decay_param,
        num_episodes=args.num_episodes or qcfg['num_episodes'],
        reward_mode=args.reward_mode, seed=args.seed,
    )
    logger = EpisodeLogger(f'results/raw_data/q_learning_{decay_type}_summary.csv')
    agent.train(logger=logger)
    logger.flush_summary()
    print(f"Q-Learning ({decay_type}) done: {len(logger.summary_rows)} episodes logged")
    return agent


def run_sarsa_lambda(env, cfg, args):
    from agents.sarsa_lambda import SarsaLambdaAgent
    scfg = cfg['sarsa_lambda']
    lam = args.lam if args.lam is not None else scfg['lambda_sweep'][0]
    agent = SarsaLambdaAgent(
        env, alpha=scfg['alpha'], gamma=scfg['gamma'], lam=lam,
        trace_type=scfg['trace_type'],
        epsilon_start=scfg['epsilon_start'], epsilon_min=scfg['epsilon_min'],
        decay_type='exponential', decay_param=scfg['exponential_decay_rate'],
        num_episodes=args.num_episodes or scfg['num_episodes'],
        reward_mode=args.reward_mode, seed=args.seed,
    )
    logger = EpisodeLogger(f'results/raw_data/sarsa_lambda_{lam}_summary.csv')
    agent.train(logger=logger)
    logger.flush_summary()
    print(f"SARSA(lambda={lam}) done: {len(logger.summary_rows)} episodes logged")
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
