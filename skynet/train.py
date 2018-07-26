import sys, os
import misc_utils
import logger_utils
import argparse
import signal
from pdqfd import PDQFDLearner
from q_network import QNetwork
import environment_creator
from emulators_coordinator import SimulatorsCoordinator

from misc_utils import boolean_flag

import logging
logger = logging.getLogger()
handler = logging.FileHandler("agent.log")
logger.setLevel(logging.DEBUG)

def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def main(args):
    if args.resume:
        conf_file = os.path.join(args.debugging_folder, 'args.json')
        assert os.path.exists(conf_file), "Could not find an args.json file in the debugging folder"
        for k, v in logger_utils.load_args(args.debugging_folder).items():
            setattr(args, k, v)

    logger.debug('Configuration: {}'.format(args))
    logger_utils.save_args(args, args.debugging_folder)
    if 'gpu' in args.device:
        agent_gpu = str(misc_utils.pick_gpu_lowest_memory())
        os.environ["CUDA_VISIBLE_DEVICES"] = agent_gpu
        logger.debug('Agent will be run on device /gpu:{}'.format(agent_gpu))

    args.random_seed = 3  # random_seed
    env_creator = environment_creator.EnvironmentCreator(args)
    args.num_actions = env_creator.num_actions
    args.state_shape = env_creator.state_shape

    import numpy as np
    # Create a set of arrays (as many as emulators) to exchange states, rewards, etc. between the agent and the emulator
    n_emulators = args.n_emulator_runners * args.n_emulators_per_emulator_runner
    variables = {"s": np.zeros((n_emulators,) + args.state_shape, dtype=np.float32),
                 "a": np.zeros((n_emulators), dtype=np.int32),  # Actions
                 "r": np.zeros((n_emulators), dtype=np.float32),  # Rewards
                 "done": np.zeros((n_emulators), dtype=np.bool)}  # Dones
    sim_coordinator = SimulatorsCoordinator(env_creator, args.n_emulators_per_emulator_runner, args.n_emulator_runners, variables)
    # Start all simulator processes
    sim_coordinator.start()

    network = QNetwork

    def network_creator(name='value_learning', learning_network=None):
            nonlocal args
            args.name = name
            return network(args, learning_network=learning_network)

    learner = PDQFDLearner(network_creator, env_creator, args, sim_coordinator)

    setup_kill_signal_handler(learner)

    logger.info('Starting training')
    learner.train()
    logger.info('Finished training')


def setup_kill_signal_handler(learner):
    main_process_pid = os.getpid()

    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logger.info('Signal ' + str(signal) + ' detected, cleaning up.')
            learner.cleanup()
            logger.info('Cleanup completed, shutting down...')
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default='FrozenLake-v0', help='Name of game', dest='game')
    parser.add_argument('--experiment_type', default='corridor', type=str, help="Class of environments to experiment with, e.g. atari, corridor, gym, etc.", dest="experiment_type")
    parser.add_argument('-d', '--device', default='cpu', type=str, help="Indicatator for whether or not the agent is trained on a gpu (Options: 'cpu', 'gpu')", dest="device")
    parser.add_argument('--rom_path', default='./atari_roms', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer to be used: Adam or Rmsprop")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.001, type=float, help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    parser.add_argument('--clip_norm', default=3.0, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument("--initial_random_steps", type=int, default=10000, help="Number of initial random steps")
    parser.add_argument('--n_steps', default=5, type=int, help="Number of steps to gain experience from before every update.", dest="n_steps")
    parser.add_argument('--arch', default='nba', help="Which network architecture to use: NBA, MLP, Deepmind's NIPS, Deepmind's NATURE", dest="arch")
    parser.add_argument('--lstm_size', default=50, type=int, help="Size of LSTM",
                        dest="lstm_size")
    parser.add_argument('--dropout', default=0.5, type=float, help="Amount of dropout",
                        dest="dropout")
    parser.add_argument('--mlp_hiddens', default='[50]', help="Hidden layers for MLP", dest="mlp_hiddens")
    boolean_flag(parser, "layer_norm", default=False, help="whether or not to use layer normalization")
    boolean_flag(parser, "double_q", default=True, help="Whether or not to use Double Q-learning")
    boolean_flag(parser, "continuous_target_update", default=True, help="Whether to update target network at fixed intervals or progressively")
    parser.add_argument('--exp_eps_segments', default='[(0, 1),(100000, 0.5),(150000,0)], 0', type=str, help="Segments for the piecewise schedule of the greedy exploration's epsilon", dest="exp_eps_segments")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('--n_emulators_per_emulator_runner', default=1, type=int, help="Number of emulators to be run by each emulator runner process. Default is 4.", dest="n_emulators_per_emulator_runner")
    parser.add_argument('--n_emulator_runners', default=1, type=int, help="Number of emulator runner processes to launch. Default is 8.", dest="n_emulator_runners")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    parser.add_argument('--clip_loss', default=1.0, type=float, help="Delta for Huber loss. Default = 1.0", dest="clip_loss_delta")
    parser.add_argument("--target_update_freq", type=int, default=10000, help="number of steps between every target network update", dest="target_update_freq")
    parser.add_argument("--target_update_tau", type=float, default=0.001,
                        help="tau for csoft, continuous target netwok update: q_target_param = tau*q_learning_param + (1-tau)*q_target_param", dest="target_update_tau")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of transitions/steps to read from the exp. replay and train with")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")


    # Prioritized experience replay
    boolean_flag(parser, "use_exp_replay", default=True, help="whether or not to use experience replay")
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6), help="replay buffer size", dest="replay_buffer_size")
    boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized_alpha", type=float, default=0.4, help="alpha parameter for prioritized replay buffer", dest="prioritized_alpha")
    parser.add_argument("--prioritized_beta0", type=float, default=0.6, help="initial value of beta parameters for prioritized replay", dest="prioritized_beta0")
    parser.add_argument("--prioritized_eps", type=float, default=1e-3, help="eps parameter for prioritized replay buffer", dest="prioritized_eps")

    parser.add_argument('--resume', help='Whether to resume training using the args.json configuration file inside debugging_folder',
                        action='store_true')

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    args.mlp_hiddens = eval(args.mlp_hiddens)

    main(args)
