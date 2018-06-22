from actor_learner import *

from schedules import LinearSchedule, PiecewiseSchedule
from replay_buffer import *

import logging
logger = logging.getLogger()

class PDQFDLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args, sim_coordinator):
        super(PDQFDLearner, self).__init__(network_creator, environment_creator, args)
        self.sim_coordinator = sim_coordinator
        self.evaluate = args.evaluate
        self.eva_env = None
        self.game = args.game
        self.double_q = args.double_q
        self.continuous_target_update = args.continuous_target_update
        self.stochastic = args.stochastic
        #self.exp_epsilon = LinearSchedule(args.max_global_steps,
        #                           initial_p=args.exp_epsilon,
        #                           final_p=0.0)
        #self.exp_epsilon = PiecewiseSchedule([(0, args.exp_epsilon), (round(args.max_global_steps/3), 0.3), (round(2*args.max_global_steps/3), 0.01)], outside_value=0.001)
        self.exp_epsilon = PiecewiseSchedule(eval(args.exp_eps_segments)[0], outside_value=eval(args.exp_eps_segments)[1])
        self.initial_random_steps = args.initial_random_steps
        self.n_emulators = self.n_emulator_runners * self.n_emulators_per_emulator_runner

        # Replay buffer
        self.use_exp_replay = args.use_exp_replay
        self.n_trajectories = round(1.0 * int(args.batch_size) / self.n_steps)
        self.replay_buffer_size = args.replay_buffer_size
        if self.use_exp_replay:
            # Create replay buffer
            self.prioritized = args.prioritized
            if self.prioritized:
                self.prioritized_alpha = args.prioritized_alpha
                self.prioritized_beta0 = args.prioritized_beta0
                self.prioritized_eps = args.prioritized_eps
                self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.state_shape,
                                                             self.prioritized_alpha, self.n_trajectories, self.n_steps,
                                                             n_emus=self.n_emulators)
                self.beta_schedule = LinearSchedule(self.max_global_steps, initial_p=self.prioritized_beta0, final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.state_shape, self.n_trajectories,
                                                  self.n_steps, n_emus=self.n_emulators)
        
        # Buffers to keep track of the last n_steps visited states
        self.states_buffer = np.ones((self.n_emulators, self.n_steps) + self.state_shape) * MASK_VALUE

        self.summaries_op = tf.summary.merge_all()
        self.counter = 0

    # The input is a tuple where each element is an array of shape (n_trajectories, n_steps) + s_shape
    # The output array has shape (n_steps * n_trajectories, n_steps) + s_shape. That is, for each trajectory,
    # each time step t in the input array is transformed into a sequence of 0 to t steps from the input array
    # and t to n masked steps.
    def prepare_input_batch(self, states):
        def masks(batch_len):
            return np.ones((batch_len, self.n_steps) + self.state_shape) * MASK_VALUE

        s_in_batch = []
        for k in range(len(states)):
            s_in_traj = masks(self.n_steps)
            for i in range(self.n_steps):
                t = np.arange(min(self.n_steps, i + 1))
                s_in_traj[i, self.n_steps -1 - t] = states[k, [i - t], :]
            s_in_batch.append(s_in_traj)

        return np.vstack(s_in_batch)

    @staticmethod
    def choose_next_actions(network, num_actions, states, session, eps, stochastic):
        network_output_q = session.run(network.output_layer_q,
            feed_dict={network.input_ph: states})

        deterministic_actions = np.argmax(network_output_q, axis=1)
        if stochastic:
            batch_size = network_output_q.shape[0]
            random_actions = np.random.randint(low=0, high=num_actions, size=batch_size)
            choose_random = np.random.uniform(low=0.0, high=1.0, size=batch_size) < eps
            stochastic_actions = np.where(choose_random, random_actions, deterministic_actions)
            action_indices = stochastic_actions
        else:
            action_indices = deterministic_actions

        return action_indices

    def __choose_next_actions(self):
        eps = self.exp_epsilon.value(self.global_step)
        return PDQFDLearner.choose_next_actions(self.network, self.num_actions, self.states_buffer, self.session, eps, self.stochastic)

    @staticmethod
    def get_target_maxq_values(target_network, next_states, session, double_q=True, learning_network=None):
        if double_q:
            [target_network_q, learning_network_q] = session.run([target_network.output_layer_q, learning_network.output_layer_q],
                                                  feed_dict={target_network.input_ph: next_states,
                                                             learning_network.input_ph: next_states})
            idx_best_action_from_learning_network = np.argmax(learning_network_q, axis=1)
            maxq_values = target_network_q[range(target_network_q.shape[0]), idx_best_action_from_learning_network]
        else:
            target_network_q = session.run(target_network.output_layer_q,
                                           feed_dict={target_network.input_ph: next_states,
                                                      learning_network.input_ph: next_states})
            maxq_values = target_network_q.max(axis=-1)

        return maxq_values

    def __get_target_maxq_values(self, next_states):
        return PDQFDLearner.get_target_maxq_values(self.target_network, next_states, self.session, double_q=self.double_q, learning_network=self.network)


    def update_target(self):
        if self.continuous_target_update:
            self.session.run(self.target_network.continuous_sync_nets)
        elif self.global_step % self.target_update_freq == 0:
            params = self.network.get_params(self.session)
            feed_dict = {}
            for i in range(len(self.target_network.params)):
                feed_dict[self.target_network.params_ph[i]] = params[i]
            self.target_network.set_params(feed_dict, self.session)


    def estimate_returns(self, next_state_maxq, rewards, dones):
        estimated_return = next_state_maxq
        done_masks = 1.0 - dones.astype(np.float32)
        y = np.zeros_like(rewards)
        for t in reversed(range(self.n_steps)):
            estimated_return = rewards[:, t] + self.gamma * estimated_return * done_masks[:, t]
            y[:, t] = estimated_return
        return y


    def train_from_experience(self):
        if self.prioritized:
            experience = self.replay_buffer.sample_nstep(self.beta_schedule.value(self.global_step))
        else:
            experience = self.replay_buffer.sample_nstep()

        (s_t, a, r, s_tp1, dones, imp_weights, idxes) = experience
        next_state_maxq = self.__get_target_maxq_values(s_tp1)
        targets = self.estimate_returns(next_state_maxq, r, dones)

        # RUN TRAIN STEP AND OBTAIN TD ERRORS
        a = np.reshape(a, -1)
        targets = np.reshape(targets, -1)
        lr = self.get_lr()
        feed_dict = {self.network.  input_ph: self.prepare_input_batch(s_t),
                        self.network.target_ph: targets,
                        self.network.importance_weights_ph: imp_weights,
                        self.network.selected_action_ph: a,
                        self.learning_rate: lr}

        _, td_errors, summaries = self.session.run(
            [self.train_step, self.network.td_error, self.summaries_op],
            feed_dict=feed_dict)

        self.summary_writer.add_summary(summaries, self.global_step)
        self.summary_writer.flush()

        self.counter += 1

        if self.prioritized:
            new_priorities = np.abs(td_errors) + self.prioritized_eps
            self.replay_buffer.update_priorities(idxes, new_priorities)

    def collect_experience(self):
        var = self.shared_variables
        for t in range(self.n_steps):
            ## Add current state to state buffer (we keep track of the last n_steps visited states to select actions)
            self.states_buffer[:, t, :] = var["s"]
            # Select next action based on sequence of states in buffer and pass on to simulators via shared_variables
            var["a"][:] = self.__choose_next_actions()

            # Start updating all environments with next_actions
            self.sim_coordinator.update_environments()
            self.sim_coordinator.wait_updated()
            # Done updating all environments, have new states, rewards and dones in shared_variables

            r = self.rescale_reward(var["r"])

            # Statistics
            self.rewards_per_step.append(var['r'].copy())

            for emu in range(self.n_emulators):
                self.replay_buffer.add(self.states_buffer[emu, t], var["a"][emu], r[emu],
                                           var["s"][emu], var["done"][emu], emu)

            for emu in np.where(var["done"] == True)[0]:
                # Reset states buffer for those emulators whose episode has ended
                for i in range(t):
                    self.states_buffer[emu, i] = MASK_VALUE

                # Statistics
                self.n_episodes += 1

        self.global_step += self.n_emulators * self.n_steps

        if self.global_step % (100 * self.n_steps) == 0:
            total_reward = np.sum(np.concatenate(self.rewards_per_step))
            self.rewards_per_step = []
            avg_reward_per_episode = total_reward / self.n_episodes
            avg_episode_length = self.global_step / self.n_episodes

            self.n_episodes = 0
            
            logger.debug("{} global steps, " 
                        "Avg. reward/episode: {:.2f}, "
                        "Avg. episode length: {:.2f}, "
                        "Epsilon: {:.2f}".format(
                                        self.global_step,
                                        avg_reward_per_episode,
                                        avg_episode_length,
                                        self.exp_epsilon.value(self.global_step)))

            stats_summary = tf.Summary(value=[
                        tf.Summary.Value(
                            tag='avg_reward_before_churn', 
                            simple_value=avg_reward_per_episode),
                        tf.Summary.Value(
                            tag='avg_episode_length', 
                            simple_value=avg_episode_length),
                    ])
            self.summary_writer.add_summary(stats_summary, self.global_step)
            self.summary_writer.flush()




    # TODO: to be fixed
    def evaluate_agent(self, msg):
        if self.evaluate:
            assert False, "Evaluate function needs to be fixed"
            if self.eva_env == None:
                self.eva_env = self.environment_creator.create_environment(-1)
            _succ_epi = evaluate(self.eva_env, self.session, self.network.output_layer_q, self.network.input_ph,
                             self.n_steps, self.state_shape,
                             visualize=False, v_func=self.network.value)
            logger.debug(
                "{}: {:.2f}%".format(msg, _succ_epi))
            perf_summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag="Performance",
                    simple_value=_succ_epi)])
            self.summary_writer.add_summary(perf_summary, self.global_step)
            self.summary_writer.flush()

    def train(self):
        """
        Main actor learner loop for parallel deep Q learning with demonstrations.
        """
        print("STARTING TRAINING")
        # Initialize networks
        self.global_step = self.init_network()
        self.update_target()
        logging.info("Synchronized learning and target networks")


        logger.debug("Resuming training from emulators at Step {}".format(self.global_step))
        self.n_episodes = self.n_emulators
        self.rewards_per_step = []

        self.sim_coordinator.update_environments()
        self.sim_coordinator.wait_updated()
        self.shared_variables = self.sim_coordinator.get_shared_variables()

        logger.debug("Shared variables accessible through simulators.")
        logger.debug("Collecting experience and training.")

        while self.global_step < self.max_global_steps:
            self.collect_experience()

            if self.global_step > self.initial_random_steps:
                self.train_from_experience()

                self.update_target()

                self.save_vars()

        self.evaluate_agent("End - Average reward over 100 episodes")

        self.cleanup()

    def cleanup(self):
        super(PDQFDLearner, self).cleanup()
        if self.n_emulators_per_emulator_runner > 0: self.sim_coordinator.stop()


