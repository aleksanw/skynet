from argparse import Namespace

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
    evaluate=False,
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
    n_steps=5,
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

custom_args = Namespace(**{**vars(default_args), **vars(Namespace(
    e = 0,
    alpha = 0.01,
))})

def main():
    import skynet.train
    skynet.train.main(custom_args)

if __name__ ==  '__main__':
    main()