import os
import tensorflow as tf
from operator import mul
from functools import reduce
import tensorflow.contrib.layers as tcl

# import matplotlib.pyplot as plt
# import random
# import numpy as np

"""Was orginally going to try and use a GAN to create new landscape images (mountains), but after struggling to find a good enough dataset, I settled
on the 'easier' task of producing human faces, but quickly learned that GANs are very hard to train and are extremlely unstable, so I was not too successful 
(was taking many hours to generate even slightly reasonable images making hyperparameter tuning very slow and time consuming)"""

# ------------------------------------- DEFINE ACTIVATION FUNCTIONS -------------------------------------------------- #


def bn_leaky_relu(inpt_tensor):
    bn = tf.layers.batch_normalization(inpt_tensor, axis=-1, training=True, scale=False, center=True)
    return tf.nn.leaky_relu(bn)


def bn_relu(inpt_tensor):
    bn = tf.layers.batch_normalization(inpt_tensor, axis=-1, training=True, scale=False, center=True)
    return tf.nn.relu(bn)


def bn_only(inpt_tensor):
    bn = tf.layers.batch_normalization(inpt_tensor, axis=-1, training=True, scale=True, center=True)
    return bn


def normalize(inpt_tensor):
    return (inpt_tensor - tf.reduce_min(inpt_tensor)) / (tf.reduce_max(inpt_tensor) - tf.reduce_min(inpt_tensor))

"""NOPE: Note: discriminator is trying to predict 1.0 for real images, 0.0 for fake images
apparently its actually slightly better to have real = 0.0, fake = 1.0 for training discriminator"""


"""   https://arxiv.org/pdf/1603.07285v1.pdf
half-padding (SAME) = doing kernel_size//2 as # of 0's added on either side, makes sure that in_size = out_size
full-padding (VALID????) = adding many 0s (k-1 0s) to the edges so the kernel will just barely touch image edges
transposed convolution (fractionally-strided convolution) = 1 -> many, normal convolution = many -> 1

transposed convolutions have direct convolutions equivalents (s' = s, k' = k, p' = (k - 1), if s = 1, p = 0, basically 
just doing full-padding of zeroes around it so you get output dimension o = i' + (k - 1), eg 3x3 kernel over a 4x4 input
goes to a 2x2 output direct (with no zero padding), so the inverse is add 3-1 = 2 zeroes around the 2x2 and then 
convolve with the 3x3 to get back the 4x4 (this time its your output though), this is so that we maintain the spatial
relationship that top-left in output corresponds to top-left in input

if direct version is with zero padding, the transpose will have fewer 0s than k - 1 case of no zero padding, then
o' = i' + ((k - 1) - 2p), and p' = (k - 1) - p, just think about it in terms of making sure that the input and output
dimensions match up, nothing else (eg. transpose of 4x4 kernel over 5x5 input padded with 2 0's (which creates 6x6 out)
is 4x4 kernel over 6x6 input with 1x1 0s)

using the matching argument, then the idea that half-padding (SAME) which has same input, output size, means that you
pad the exact same amount for both transpose and normal, and also that if you do full padding as normal, then the
transpose has no padding (eg. transpose of 3x3 kernel over 5x5 input with full padding creates 7x7 output, is a 3x3 
kernel over 7x7 input with no padding -> 5x5 output)

fractionally-strided convolutions come in when you have non-unit strides as normal, then you want to transpose it (if
stride = 3, then you want to do stride of "1/3") to make this possible, you put zeros between stuff in the input, making
kernel move at a slower pace with unit strides, its easy if i - k is a nice multiple of s and p = 0, then the transpose
is ~i' = size of input i' stretched by adding s - 1 zeroes between inputs, p' = k - 1, s' = 1, k' = k, and 
o' = s(i' - 1) + k (eg. 3x3 kernel over 5x5 input with 2x2 stride (-> 2x2 out) transpose is 3x3 kernel over 2x2 input 
with 1 zero between each element, padded by 2x2 zeros, -> 5x5 out)

most general case of non-unit, zero-padded transpose where i - k + 2p isn't multiple of s, is the same as the one above 
just  o' -= 2p, as in unit-stride case, but you may have to add some extra zeros of (i + 2p - k) mod s to one side
then you get finally o' = s(i' - 1) + ((i + 2p - k) mod s) + k - 2p
"""

# --------------------------------------------- TRAINING PARAMETERS --------------------------------------------------#
PZ_SIZE = 8
BETA1 = 0.5
BETA2 = 0.9
EPOCHS = 30000
BENCHMARK_TIME = 15
BATCH_SIZE = 16
N_CRITIC = 5
LAMBDA = 10.0
IMG_SIZE = 128
LEARNING_RATE_GENERATOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
CLIP = 5.0
# --------------------------------------------- MODEL PARAMETERS ---------------------------------------------------#

LOG_PATH = "/tmp/tensorflow_logs/GAN_105"
UP_FILTER_SIZES = [4, 4, 16]
# index 0 and 1 represent scaled up feature map size, index 2 represents number of feature maps to make
UP_CONV_SIZES = [[4, 4, 1024], [16, 16, 512], [64, 64, 128], [IMG_SIZE, IMG_SIZE, 16]]
# tanh cuz WGAN != probability, should be called "value function", call it from "discriminator" -> "critic"
UP_FUNCS = [bn_relu, bn_relu, bn_relu]

DOWN_FILTER_SIZES = [4, 4, 4, 4]
# index 0 and 1 represent next feature map size (roughly, done using //) and index 2 is feature maps
DOWN_CONV_SIZES = [[IMG_SIZE, IMG_SIZE, 3], [32, 32, 64], [16, 16, 128], [8, 8, 256], [4, 4, 512]]
DOWN_FUNCS = [tf.nn.leaky_relu, bn_leaky_relu, bn_leaky_relu, bn_leaky_relu]

assert(len(UP_FILTER_SIZES) == len(UP_FUNCS) == len(UP_CONV_SIZES) - 1)
assert(len(DOWN_FILTER_SIZES) == len(DOWN_FUNCS) == len(DOWN_CONV_SIZES) - 1)
# ------------------------------------------------- LOAD DATA --------------------------------------------------------#
abs_path = os.getcwd()
img_list = []
for root_dir, nothing, imgs in os.walk("datasetts\\lfw"):
    img_list.extend([os.path.join(root_dir, img) for img in imgs])

total_len = len(img_list)


def to_image(filename):
    img_str = tf.read_file(filename)
    img_decode = tf.expand_dims(tf.image.decode_jpeg(img_str, channels=3), dim=0)
    img_resize = tf.image.resize_bilinear(img_decode, [IMG_SIZE, IMG_SIZE], align_corners=True)
    img = tf.cast(img_resize, tf.float32) / 255.
    return tf.squeeze(img, 0)


img_dataset = tf.data.Dataset.from_tensor_slices(img_list[:1])
img_dataset = img_dataset.shuffle(buffer_size=total_len)
img_dataset = img_dataset.map(to_image)
img_dataset = img_dataset.repeat().batch(BATCH_SIZE)  # means we can continuously iterate over it
iterator = img_dataset.make_one_shot_iterator()
next_faces = iterator.get_next()

# ------------------------------------------------- DEFINE MODEL -----------------------------------------------------#

g_step_tensor = tf.Variable(0, trainable=False)
d_step_tensor = tf.Variable(0, trainable=False)


def get_theta(name, shape):
    theta = tf.get_variable(name, shape=shape, trainable=True,
                            initializer=tf.random_normal_initializer(stddev=0.02),
                            constraint=lambda x: tf.clip_by_value(x, -CLIP, CLIP))
    return theta


def convolve_up(input_tensor, convolver, s1, s2, upsize, act):
    # resized = tf.image.resize_images(input_tensor, upsize, align_corners=True)
    up_conv = tf.nn.conv2d_transpose(input_tensor, convolver, strides=[1, s1, s2, 1], padding="SAME", output_shape=upsize)
    up_conv_a = act(up_conv)
    return up_conv_a


def convolve_down(input_tensor, convolver, s1, s2, act):
    down_conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, s1, s2, 1], padding="SAME")
    down_conv_a = act(down_conv)
    return down_conv_a


def fully_connected(input_tensor, fc, act_func):
    reshaped = tf.reshape(input_tensor, [BATCH_SIZE, tf.shape(fc)[0]])
    fc_logits = act_func(tf.matmul(reshaped, fc))
    return fc_logits


def generator_model(pz_vectors):
    with tf.variable_scope("up_layers"):
        fc = tcl.fully_connected(pz_vectors, reduce(mul, UP_CONV_SIZES[0]), activation_fn=tf.identity)
        fc_r = tf.reshape(fc, tf.stack([BATCH_SIZE, *UP_CONV_SIZES[0]]))
        conv = bn_relu(fc_r)
        for filt, dim, prev_dim, act in zip(UP_FILTER_SIZES, UP_CONV_SIZES[1:], UP_CONV_SIZES[:-1], UP_FUNCS):
            conv = tcl.conv2d_transpose(
                conv, dim[2], [filt, filt], [dim[0]//prev_dim[0], dim[1]//prev_dim[1]],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=act
            )
        final_conv = tcl.conv2d(
            conv, 3, [24, 24], [1, 1],
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.tanh
        )
    return (final_conv)


def critic_model(x, reuse=True):
    with tf.variable_scope("down_layers") as vs:
        if reuse:
            vs.reuse_variables()
        conv = tf.reshape(x, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
        for filt, prev_dim, dim, act in zip(DOWN_FILTER_SIZES, DOWN_CONV_SIZES[:-1], DOWN_CONV_SIZES[1:], DOWN_FUNCS):
            conv = tcl.conv2d(
                conv, dim[2], [filt, filt], [prev_dim[0]//dim[0], prev_dim[1]//dim[1]],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=act
            )

        final_conv = tcl.flatten(conv)
        fc = tcl.fully_connected(final_conv, 1, activation_fn=tf.identity)
        return fc


# ------------------------------------ DEFINE OPTIMIZATION AND LOSS OPERATIONS ---------------------------------------#

noise = tf.random_uniform([BATCH_SIZE, PZ_SIZE], -1., 1.)
epsilon = tf.random_uniform([BATCH_SIZE, 1, 1, 1], 0, 1)

g_img = generator_model(noise)  # G(pz_noise)
d_real_logits = critic_model(next_faces, reuse=False)  # to be used in critic optimization
d_fake_logits = critic_model(g_img)  # to be used in critic optimization

# э * x + (1 - э) * x~ -> x~ + э * (x - x~)
combined_img = g_img + epsilon * (next_faces - g_img)
d_combined_logits = critic_model(combined_img)  # to be used in grad penalty

# computes E[norm(∇D(combined_img)) − 1)ˆ2]
d_grads = tf.gradients(d_combined_logits, combined_img)[0]
grad_norms = tf.sqrt(tf.reduce_sum(tf.square(d_grads), axis=1))
grad_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
# actual critic optimization loss, min{E[D(fake)] - E[D(real)] + λ * E[(norm(grad) -  1) ** 2]}
critic_loss = tf.reduce_mean(d_real_logits) - tf.reduce_mean(d_fake_logits) + LAMBDA * grad_penalty

# tries to minimize D(G(pz_noise)), so that the critic thinks its a real image being produced
generator_loss = tf.reduce_mean(d_fake_logits)


d_vars = [w for w in tf.trainable_variables() if "down" in w.name]
g_vars = [theta for theta in tf.trainable_variables() if 'up' in theta.name]

generator_gradients = [tf.reduce_mean(layer1) + tf.reduce_mean(layer2) for layer1, layer2 in
                       zip(tf.gradients(generator_loss, g_vars)[:-1], tf.gradients(generator_loss, g_vars)[1:])]

critic_gradients = [tf.reduce_mean(layer1) + tf.reduce_mean(layer2) for layer1, layer2 in
                    zip(tf.gradients(critic_loss, d_vars)[:-1], tf.gradients(critic_loss, d_vars)[1:])]


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    generator_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_GENERATOR, beta1=BETA1,
                                           beta2=BETA2).minimize(
        generator_loss, var_list=g_vars,
        global_step=g_step_tensor)

    critic_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_CRITIC, beta1=BETA1,
                                        beta2=BETA2).minimize(
        critic_loss, var_list=d_vars,
        global_step=d_step_tensor)

with tf.name_scope("summary_op"):
    # summaries when you call the generator optimization operation, to see some progress
    g_img_summaries = [tf.summary.image("generated_img", g_img),
                       # tf.summary.image("combined_img", combined_img),
                       tf.summary.image("real_img", next_faces)]
    g_tensor_summaries = [tf.summary.histogram("fake_logits", d_fake_logits),
                          tf.summary.histogram("real_logits", d_real_logits),
                          tf.summary.scalar("generator_loss", generator_loss),
                          tf.summary.histogram("generator_values", g_img),
                          tf.summary.histogram("actual_image", next_faces)]
    d_tensor_summaries = [tf.summary.scalar('critic_loss', critic_loss),
                          tf.summary.scalar("grad_penalty", grad_penalty),
                          tf.summary.histogram('critic_grads', d_grads)]
    grad_summaries = []
    for idx, (g_grad, d_grad) in enumerate(zip(generator_gradients, critic_gradients)):
        grad_summaries.append(tf.summary.scalar(f"gen-{idx}-grad", g_grad))
        grad_summaries.append(tf.summary.scalar(f"dis-{idx}-grad", d_grad))

    generator_summary = tf.summary.merge(g_img_summaries + g_tensor_summaries + grad_summaries)
    critic_summary = tf.summary.merge(d_tensor_summaries)

# -------------------------------------------- DEFINE TRAINING LOOP ---------------------------------------------------#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    loader = tf.train.Saver()
    writer = tf.summary.FileWriter(LOG_PATH)
    try:
        loader.restore(sess, tf.train.latest_checkpoint('face_GAN'))
    except ValueError:
        print("No model found, initializing default model")
        grapher = tf.summary.FileWriter(LOG_PATH, sess.graph)
        grapher.add_summary(tf.Summary(), 0)
    save_mark = 1
    for ep in range(EPOCHS):
        for t in range(N_CRITIC):
            if t == 0:
                _, d_sum = sess.run([critic_opt, critic_summary])
                writer.add_summary(d_sum, tf.train.global_step(sess, d_step_tensor))
            else:
                sess.run(critic_opt)
            print(ep, ":", t)
        ____, gen_sum = sess.run([generator_opt, generator_summary])
        writer.add_summary(gen_sum, tf.train.global_step(sess, g_step_tensor))
        # if ep == save_mark:
        #     saver.save(sess, f"face_GAN\\{tf.train.global_step(sess, g_step_tensor)}")
        #     save_mark += BENCHMARK_TIME
