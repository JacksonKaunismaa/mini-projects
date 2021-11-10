import tensorflow as tf
import numpy as np

# based on https://arxiv.org/pdf/1707.04585.pdf

LARGE_FILTER = 3
BOTTLENECK = 64
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
LEARNING_RATE = 0.001
CHANNELS = [32, 32, 64, 128]
UNITS = [9, 9, 9]



def get_weight(name, shape):
    weight = tf.get_variable(name, shape=shape, trainable=True,
                            initializer=tf.variance_scaling_initializer())
    return weight

def get_adam(name, shape):
    adam_param = tf.get_variable(name, shape=shape, trainable=False,
                                 initializer=tf.zeros_initializer())

class RevNet():
    self.graph() = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
        self.f_weights = {}
        self.f_momentum = {}
        self.f_velocity = {}

        self.g_weights = {}
        self.g_momentum = {}
        self.g_velocity = {}
        for i in range(LAYERS):
            f_name = f"layer-{layer_idx}-F"
            g_name = f"layer-{layer_idx}-G"
            self.f_weights[f_name][0] = "i have no idea how to finish this project so i give up and move to more important stuff" 

def convolve(input_tensor, convolver, act):
    down_conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, 1, 1, 1], padding="SAME")
    down_conv_a = act(down_conv)

def bn_relu(inpt_tensor):
    bn = tf.layers.batch_normalization(inpt_tensor, axis=-1, training=True, scale=False, center=True)
    return tf.nn.relu(bn)


def bottleneck_network(layer_name, inpt):
    # essentially F or G in the RevNet paper (2017), no residual connections, CNN but with bottleneck to decr num channels
    inpt_shapes = tf.shape(inpt)
    decr_channels, conv, incr_channels = get_bottleneck_weights(layer_name)
    with tf.variable_scope(layer_name):
        activated = bn_relu(inpt)
        decr_log = tf.nn.conv2d(activated, decr_channels, strides=[1,1,1,1], padding="SAME")
        decr_act = bn_relu(decr_log)
        conv_log = tf.nn.conv2d(decr_act, conv, strides=[1,1,1,1], padding="SAME")
        conv_act = bn_relu(conv_log)
        incr_log = tf.nn.conv2d(conv_act, incr_channels, strides=[1,1,1,1], padding="SAME")
        return incr_log

def forward_reversible_block(layer_idx,  inpt):
    # again using variable names as in the RevNet 2017 paper
    x1, x2 = tf.split(inpt, 2, axis=-1)    # split in half channel-wise
    z1 = x1 + bottleneck_network(f"layer-{layer_idx}-F", x2)
    y2 = x2 + bottleneck_network(f"layer-{layer_idx}-G", z1)
    y1 = z1
    return tf.concat((y1, y2), axis=-1)

def get_bottleneck_weights(layer_name):
    with tf.variable_scope(layer_name):
        with tf.name_scope("weights"):
            decr_channels = get_weight("decr_channels", [1, 1, inpt_shapes[3], BOTTLENECK])
            conv = get_weight("conv", [LARGE_FILTER, LARGE_FILTER, BOTTLENECK, BOTTLENECK])
            incr_channels = get_weight("incr_channels", [1, 1, BOTTLENECK, inpt_shapes[3]])
        return decr_channels, conv, incr_channels

def apply_updates(layer_name, total_derivatives, weights, iter_t):
    with tf.variable_scope(layer_name):
        for i, (weight_total, actual_weight) in enumerate(zip(total_derivatives, weights)):
            momentum_t = BETA1 * get_adam(f"{layer_name}-momentum_t-{i}", tf.shape(weight_total)) + (1 - BETA1) * weight_total
            velocity_t = BETA2 * get_adam(f"{layer_name}-velocity_t-{i}", tf.shape(weight_total)) + (1 - BETA2) * tf.square(weight_total)
            actual_weight -= LEARNING_RATE * momentum_t / (tf.sqrt(velocity_t) + EPSILON)

def reverse_reversible_block(layer_idx, inpt, total_derivatives, update_weights=True):
    y1, y2 = tf.split(inpt, 2, axis=-1)
    y1_total, y2_total = tf.split(total_derivatives, 2, axis=-1)
    z1 = y1
    g_out = bottleneck_network(g_name, z1)
    x2 = y2 - g_out
    f_put = bottleneck_network(f_name, x2)
    x1 = z1 - f_out
    z1_total = y1_total + tf.gradients(g_out, z1) * y2_total
    x2_total = y2_total + tf.gradients(f_out, x2) * z1_total
    x1_total = z1_total
    if update_weights:
        f_weights = get_bottleneck_weights(f_name)
        g_weights = get_bottleneck_weights(g_name)
        wf_totals = [tf.gradients(f_out, f_weight)*z1_total for f_weight in f_weights]
        wg_totals = [tf.gradients(g_out, g_weight)*y2_total for g_weight in g_weights]
    return tf.concat((x1, x2), axis=-1), tf.concat((x1_total, x2_total), axis=-1)




test_name = "./datasets/pkl/test.tfr"
train_name = "./datasets/pkl/train.tfr"

raw_train = tf.data.TFRecordDataset(train_name)
raw_test = tf.data.TFRecordDataset(test_name)

tfr_description = {"img": tf.FixedLenFeature([], tf.float32, default_value=0.0),
                   "lbl": tf.FixedLenFeature([], tf.int64, default_value=0),
                   "name": tf.FixedLenFeature([], tf.string, default_value='')}

def _parse_tfr(proto):
    return tf.parse_single_example(proto, tfr_description)

train_set = raw_train.map(_parse_tfr)
test_set = raw_test.map(_parse_tfr)

