from argparse import Namespace

import logging
log = logging.getLogger(__name__)

def namespace_based_on(base_namespace, **kwargs):
    return Namespace(**{**vars(base_namespace), **vars(Namespace(**kwargs))})


default_args = Namespace(
    alpha=0.99,
    arch='nba',
    batch_size=256,
    clip_loss_delta=1.0,
    clip_norm=3.0,
    clip_norm_type='global',
    continuous_target_update=True,
    debugging_folder='logs/',
    device='cpu',
    double_q=True,
    dropout=0.5,
    e=0.1,
    exp_eps_segments='[(0, 1),(10000, 0.5),(15000,0)], 0',
    experiment_type='corridor',
    game='FrozenLake-v0',
    gamma=0.99,
    initial_lr=0.001,
    initial_random_steps=10000,
    layer_norm=False,
    lr_annealing_steps=80000000,
    lstm_size=50,
    max_global_steps=80000000,
    mlp_hiddens=[50],
    n_emulator_runners=8,
    n_emulators_per_emulator_runner=4,
    n_steps=5,  # length of trajectories to be samples from replay buffer while n-step sampling.
    optimizer='adam',
    prioritized=True,
    prioritized_alpha=0.4,
    prioritized_beta0=0.6,
    prioritized_eps=0.001,
    random_start=True,
    replay_buffer_size=1000000,
    resume=False,
    rom_path='./atari_roms',
    single_life_episodes=False,
    stochastic=True,
    target_update_freq=10000,
    target_update_tau=0.001,
    use_exp_replay=True,
    visualize=False,
)


custom_args = namespace_based_on(
    default_args,
    exp_eps_segments='[(0, 1),(10000, 0.5),(100000,0)], 0',
    initial_random_steps=10000,
    n_emulator_runners=1,
    n_emulators_per_emulator_runner=2,
    batch_size=32,
    max_global_steps=300000,  # Empirically, seems to be enough
    target_update_freq=2000,
    target_update_tau=0.05,
)


def main():
    #import warnings
    #warnings.simplefilter('error')
    # Tensorflow makes use of deprecated module imp. Squelch that warning.
    #warnings.filterwarnings('ignore', '.*imp module.*',)
    logging.basicConfig(level=logging.INFO)
    log.info(f"Running skynet.train.main with args {custom_args}")
    import skynet.train
    skynet.train.main(custom_args)


if __name__ ==  '__main__':
    main()