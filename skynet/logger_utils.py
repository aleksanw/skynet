import os
import json



def load_args(path):
    conf_file = os.path.join(path, 'args.json')
    if not os.path.exists(conf_file):
        return {}
    with open(conf_file, 'r') as f:
        return json.load(f)


def save_args(args, folder, file_name='args.json'):
    args = vars(args)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, file_name), 'w') as f:
        return json.dump(args, f)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    import tensorflow as tf
    with tf.name_scope('summaries'):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
