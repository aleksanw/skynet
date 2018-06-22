#from emulator_runner import EmulatorRunnerWorker
#from runners import Runner
import q_network
import numpy as np
from multiprocessing import Process
import tensorflow as tf
from logger_utils import variable_summaries
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


CHECKPOINT_INTERVAL = 1000000
 

class ActorLearner(Process):
    
    def __init__(self, network_creator, environment_creator, args):
        
        super(ActorLearner, self).__init__()

        tf.reset_default_graph()

        self.global_step = 0

        self.environment_creator = environment_creator
        self.network_creator = network_creator

        self.n_steps = args.n_steps
        self.state_shape = args.state_shape
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps
        self.n_emulators_per_emulator_runner = args.n_emulators_per_emulator_runner
        self.n_emulator_runners = args.n_emulator_runners
        self.device = args.device
        self.debugging_folder = args.debugging_folder
        self.network_checkpoint_folder = os.path.join(self.debugging_folder, 'checkpoints/')
        self.optimizer_checkpoint_folder = os.path.join(self.debugging_folder, 'optimizer_checkpoints/')
        self.last_saving_step = 0
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.debugging_folder, 'tf'))

        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma
        self.game = args.game

        self.arch = args.arch
        self.network = network_creator(name='value_learning')
        self.target_network = network_creator(name='value_target', learning_network=self.network)
        self.target_update_freq = args.target_update_freq

        self.train_step, flat_raw_gradients, flat_clipped_gradients, global_norm, self.learning_rate = q_network.train_operation(
            self.network, args)

        config = tf.ConfigProto()
        if 'gpu' in self.device:
            logger.debug('Dynamic gpu mem allocation')
            config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.network_saver = tf.train.Saver()

        # Summaries
        variable_summaries(flat_raw_gradients, 'raw_gradients')
        variable_summaries(flat_clipped_gradients, 'clipped_gradients')
        tf.summary.scalar('global_norm', global_norm)
        tf.summary.scalar("TD_loss", self.network.td_loss)

    def save_vars(self, force=False):
        if force or self.global_step - self.last_saving_step >= CHECKPOINT_INTERVAL:
            self.last_saving_step = self.global_step
            self.network_saver.save(self.session, self.network_checkpoint_folder, global_step=self.last_saving_step)

    def rescale_reward(self, reward, type='log'):
        if type == 'log':
            reward = np.sign(reward) * np.log(1 + np.abs(reward))
        elif type == 'normalize':
            reward = 1.0 * reward / self.max_reward
        elif type == 'clip':
            """ Clip immediate reward """
            assert False, "Fix reward clipping to account for arrays"
            if reward > 1.0:
                reward = np.ones_like(reward)
            elif reward < -1.0:
                reward = -1.0*np.ones_like(reward)
        return reward

    def init_network(self):
        import os
        if not os.path.exists(self.network_checkpoint_folder):
            os.makedirs(self.network_checkpoint_folder)
        if not os.path.exists(self.optimizer_checkpoint_folder):
            os.makedirs(self.optimizer_checkpoint_folder)

        # Since we only save and restore network trainable variables and optimizer variables, we always need to
        # initialize the other variables
        self.session.run(tf.global_variables_initializer())

        # This should restore both the local/learning network and the target network
        last_saving_step = self.network.init(self.network_checkpoint_folder, self.network_saver, self.session)

        return last_saving_step

    def get_lr(self, decay=False):
        if decay:
            if self.global_step <= self.lr_annealing_steps:
                return self.initial_lr - (self.global_step * self.initial_lr / self.lr_annealing_steps)
            else:
                return 0.0
        else:
            return self.initial_lr

    def cleanup(self):
        logger.info('Saving vars before shutting down...')
        self.save_vars(True)
        logger.info('Vars saved.')
        self.session.close()

