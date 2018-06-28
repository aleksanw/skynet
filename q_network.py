import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Flatten, Input, Lambda, Activation, LSTM, Masking, Bidirectional, Add, TimeDistributed
#from tensorflow.python.keras import regularizers
from tensorflow.contrib.layers import layer_norm
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

MASK_VALUE = -100.

class QNetwork(object):
    def __init__(self, conf, learning_network=None):
        self.name = conf.name
        self.num_actions = conf.num_actions
        self.clip_loss_delta = conf.clip_loss_delta
        self.clip_norm = conf.clip_norm
        self.clip_norm_type = conf.clip_norm_type
        self.device = conf.device

        self.dueling_type = 'avg'

        with tf.device(self.device):
            with tf.variable_scope(self.name):
                self.input_ph = tf.placeholder(shape=(None, conf.n_steps) + conf.state_shape, dtype=tf.float32, name='input')
                input = Input(tensor=self.input_ph)
                hidden = Masking(mask_value=MASK_VALUE)(input)
                #hidden = Flatten()(input)
                for nodes in conf.mlp_hiddens:
                    hidden = Dense(nodes)(hidden)
                    if conf.layer_norm:
                        # hidden = Lambda(lambda x: layer_norm(x))(hidden, center=True, scale=True)
                        hidden = Lambda(layer_norm, center=True, scale=True)(hidden)
                    hidden = Activation('relu')(hidden)

                hidden = LSTM(conf.lstm_size)(hidden)
                self.output_layer_q = Dense(self.num_actions)(hidden)

                # Create auxiliary model to apply the Dueling architecture via TimeDistributed to each
                # timestep of the BLSTM output
                #dueling_inp = Input(shape=(1, 2*lstm_nodes))
                #advantage = Dense(self.num_actions)(dueling_inp)
                #value = Dense(1)(dueling_inp)
                #adv_minus_adv_mean = Lambda(lambda adv: tf.subtract(adv, tf.reduce_mean(
                #    adv,
                #    axis=1,
                #    keepdims=True)))(advantage)
                #dueling_out = Add()([value, adv_minus_adv_mean])
                #dueling = Model(inputs=dueling_inp, outputs=dueling_out)
                #self.output_layer_q = TimeDistributed(dueling)(hidden)

                #self.model = Model(input, self.output_layer_q)
                self.params = [v for v in tf.trainable_variables() if self.name in v.name] #self.model.trainable_weights

                if "value_learning" in self.name:  # learning network
                    self.selected_action_ph = tf.placeholder("int32", [None],#, self.num_actions],
                                                             name="selected_action")
                    selected_action = tf.one_hot(self.selected_action_ph, self.num_actions, dtype=tf.float32)

                    self.target_ph = tf.placeholder("float32", [None], name='target')

                    self.output_selected_action = tf.reduce_sum(
                        tf.multiply(self.output_layer_q, selected_action),
                        reduction_indices=1)

                    # importance weights for every element of the batch (gradient is multiplied
                    # by the importance weight)
                    self.importance_weights_ph = tf.placeholder(tf.float32, [None], name="importance_weight")

                    # TD loss (Huber loss)
                    delta = self.clip_loss_delta
                    self.td_error = tf.subtract(self.target_ph, self.output_selected_action)
                    self.td_loss = tf.where(tf.abs(self.td_error) < delta,
                                       tf.square(self.td_error) * 0.5,
                                       delta * (tf.abs(self.td_error) - 0.5 * delta))
                    self.weighted_td_loss = tf.reduce_mean(self.importance_weights_ph * self.td_loss)

                    # if self.clip_loss_delta > 0:
                    #     quadratic_part = tf.minimum(tf.abs(diff),
                    #                                 tf.constant(self.clip_loss_delta))
                    #     linear_part = tf.sub(tf.abs(diff), quadratic_part)
                    #     td_loss = tf.add(tf.nn.l2_loss(quadratic_part),
                    #                      tf.mul(tf.constant(self.clip_loss_delta), linear_part))
                    # else:
                    #     #td_loss = tf.nn.l2_loss(diff)
                    #     td_loss = tf.reduce_mean(tf.square(diff))


                    self.loss = self.weighted_td_loss

                elif "value_target" in self.name:
                    if conf.continuous_target_update:
                        assert learning_network is not None, "Need to pass the learning network as argument when creating the target network"
                        tau = tf.constant(conf.target_update_tau, dtype=np.float32)
                        self.continuous_sync_nets = []
                        for i in range(len(learning_network.params)):
                            self.continuous_sync_nets.append(
                                self.params[i].assign(
                                    tf.multiply(learning_network.params[i].value(), tau) +
                                    tf.multiply(self.params[i], tf.subtract(tf.constant(1.0),tau))))
                    else:
                        self.params_ph = []
                        for p in self.params:
                            self.params_ph.append(tf.placeholder(tf.float32,
                                                                 shape=p.get_shape(),
                                                                 name='params_to_sync'))

                        self.discrete_sync_nets = []
                        for i in range(len(self.params)):
                            self.discrete_sync_nets.append(
                                self.params[i].assign(self.params_ph[i]))

    def get_params(self, session):
        with tf.device(self.device):
            with tf.name_scope(self.name):
                param_values = session.run(self.params)
                return param_values

    def set_params(self, feed_dict, session):
        with tf.device(self.device):
            with tf.name_scope(self.name):
                session.run(self.discrete_sync_nets, feed_dict=feed_dict)

    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device(self.device):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            if path is None:
                # We start from scartch. All the tensorflow graph variables have been already initialized before
                # coming here. Here we just synchronize the learning and target networks
                logging.info('Initialized all variables.')
                #session.run(tf.global_variables_initializer())
            else:
                #session.run(tf.global_variables_initializer())
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-') + 1:])
                logging.info('Restored network variables from previous run')
        return last_saving_step


def train_operation(network, args):
    # Optimizer
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer_variable_names = 'OptimizerVariables'
    if args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate, name=optimizer_variable_names)
    else:
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=args.alpha, epsilon=args.e,
                                              name=optimizer_variable_names)

    grads_and_vars = optimizer.compute_gradients(
        network.loss, network.params)
    flat_raw_gradients = tf.concat(
        [tf.reshape(g, [-1]) for g, v in grads_and_vars], axis=0)

    # This is not really an operation, but a list of gradient Tensors.
    # When calling run() on it, the value of those Tensors
    # (i.e., of the gradients) will be calculated
    if args.clip_norm_type == 'ignore':
        # Unclipped gradients
        global_norm = tf.global_norm(
            [g for g, v in grads_and_vars], name='global_norm')
    elif args.clip_norm_type == 'global':
        # Clip network grads by network norm
        gradients_n_norm = tf.clip_by_global_norm(
            [g for g, v in grads_and_vars], args.clip_norm)
        global_norm = tf.identity(gradients_n_norm[1], name='global_norm')
        grads_and_vars = list(
            zip(gradients_n_norm[0], [v for g, v in grads_and_vars]))
    elif args.clip_norm_type == 'local':
        # Clip layer grads by layer norm
        gradients = [tf.clip_by_norm(
            g, args.clip_norm) for g in grads_and_vars]
        grads_and_vars = list(
            zip(gradients, [v for g, v in grads_and_vars]))
        global_norm = tf.global_norm(
            [g for g, v in grads_and_vars], name='global_norm')
    else:
        raise Exception('Norm type not recognized')

    flat_clipped_gradients = tf.concat(
        [tf.reshape(g, [-1]) for g, v in grads_and_vars], axis=0)

    train = optimizer.apply_gradients(grads_and_vars)

    return train, flat_raw_gradients, flat_clipped_gradients, global_norm, learning_rate