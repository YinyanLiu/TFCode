#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env: python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     :
# @File     : gputest.py
# @Time     : 31/07/2018, 4:26 PM
"""
import os

import tensorflow as tf

# set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# set the visible_devices
os.environ['CUDA_VISIBLE_DEVICES'] = '12, 13, 14, 15'
# GPU list
N_GPU = 4 # GPU number

# define parameters of neural network
BATCH_SIZE = 100*N_GPU
LEARNING_RATE = 0.001
EPOCHS_NUM = 1000
NUM_THREADS = 10

# define the path of log message and model
DATA_DIR = 'data/tmp/'
LOG_DIR = 'data/tmp/log'
DATA_PATH = 'data/test_data.tfrecord'

# get train data
def _parse_function(example_proto):
    dics = {
        'sample': tf.FixedLenFeature([5], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    parsed_example['sample'] = tf.cast(parsed_example['sample'], tf.float32)
    parsed_example['label'] = tf.cast(parsed_example['label'], tf.float32)

    return parsed_example
def _get_data(tfrecord_path = DATA_PATH, num_threads = NUM_THREADS, num_epochs = EPOCHS_NUM, batch_size = BATCH_SIZE, num_gpu = N_GPU):
    with tf.variable_scope('input_data'):
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        new_dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
        shuffle_dataset = new_dataset.shuffle(buffer_size=10000)
        repeat_dataset = shuffle_dataset.repeat(num_epochs)
        batch_dataset = repeat_dataset.batch(batch_size=batch_size)
        iterator = batch_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        x_split = tf.split(next_element['sample'], num_gpu)
        y_split = tf.split(next_element['label'], num_gpu)
    return x_split, y_split

def weight_bias_variable(weight_shape, bias_shape):
    weight = tf.get_variable('weight', weight_shape, initializer=tf.random_normal_initializer(mean=0, stddev=1))
    bias = tf.get_variable('bias', bias_shape, initializer=tf.random_normal_initializer(mean=0, stddev=1))
    return weight, bias


def hidden_layer(x_data, input_dim, output_dim, layer_name):
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        weight, bias = weight_bias_variable([input_dim, output_dim], [output_dim])
        # calculation output
        y_hidden = tf.nn.relu(tf.matmul(x_data, weight) + bias)
        tf.summary.histogram('weight', weight)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('y_hidden', y_hidden)
    return y_hidden

def output_grads(y_hidden, y_label, input_dim, output_dim):
    with tf.variable_scope('out_layer', reuse=tf.AUTO_REUSE):
        weight, bias = weight_bias_variable([input_dim, output_dim], [output_dim])
        tf.summary.histogram('bias', bias)
        y_out = tf.matmul(y_hidden, weight) + bias
        y_out = tf.reshape(y_out, [-1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=y_label)
        loss_mean = tf.reduce_mean(loss, 0)
        tf.summary.scalar('loss', loss_mean)
        grads = opt.compute_gradients(loss_mean)
    return loss_mean, grads

# calculate gradient
def average_gradients(tower_grads):
    avg_grads = []

    # list all the gradient obtained from different GPU
    # grad_and_vars represents gradient of w1, b1, w2, b2 of different gpu respectively
    for grad_and_vars in zip(*tower_grads): # 第一个循环中的grad_and_vars是((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1)...
        # 第二次循环是第二个参数：((grad1_gpu0,var1_gpu0),(grad1_gpu1, var1_gpu1)......
        # calculate average gradients
        grads = []
        for g, _ in grad_and_vars: # different gpu # 不同GPU下同一参数的梯度值
            expanded_g = tf.expand_dims(g, 0)  # expand one dimension (5, 10) to (1, 5, 10)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0) # for 4 gpu, 4 (1, 5, 10) will be (4, 5, 10),concat the first dimension
        grad = tf.reduce_mean(grad, 0) # calculate average by the first dimension
        # print('grad: ', grad)

        v = grad_and_vars[0][1] # 变量名，因为多个GPU参数是共享的，只取第一个[0]就行
        # print('v',v)
        grad_and_var = (grad, v)

        print('grad_and_var: ', grad_and_var)
        # corresponding variables and gradients
        avg_grads.append(grad_and_var)

    return avg_grads

# get samples and labels
with tf.name_scope('input_data'):
    x_split, y_split = _get_data()
# set optimizer
opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
tower_grads = []
for i in range(N_GPU):
    with tf.device("/gpu:%d" % i):
        with tf.name_scope('GPU_%d' %i) as scope:
            y_hidden = hidden_layer(x_split[i], input_dim=5, output_dim=10, layer_name='hidden1')
            loss_mean, grads = output_grads(y_hidden, y_label=y_split[i], input_dim=10, output_dim=1)
            tower_grads.append(grads)
with tf.name_scope('update_parameters'):
    # get average gradient
    grads = average_gradients(tower_grads)
    for i in range(len(grads)):
        tf.summary.histogram('gradients/'+grads[i][1].name, grads[i][0])
    # update parameters。
    apply_gradient_op = opt.apply_gradients(grads)


init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.allow_soft_placement = True
config.log_device_placement = False
with tf.Session(config=config) as sess:

    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('data/tfboard', sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for step in range(1000):
        sess.run(apply_gradient_op)
        summary = sess.run(merged)
        writer.add_summary(summary, step)
    writer.close()
    coord.request_stop()
    coord.join(threads)

