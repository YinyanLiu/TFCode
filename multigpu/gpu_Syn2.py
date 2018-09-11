#!/usr/bin/env python
# _*_coding:utf-8 _*_

"""
# Env       : python3.6
# @Author   : Liuyinyan
# @Contact  : yinyan.liu@rokid.com
# @Site     : 
# @File     : gpu_Syn2.py
# @Time     : 16/08/2018, 10:04 AM
"""
import tensorflow as tf
from tensorflow.python.client import device_lib
import os

# There are different methods to control logging output from tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# under Linux environment, you can use $ export TF_CPP_MIN_LOG_LEVEL=3 before you xx.py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#################  获取当前设备上的所有GPU ##################
def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))
    return gpu_num

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
MODEL_SAVE_PATH = 'data/tmp/logs_and_models/'
MODEL_NAME = 'model.ckpt'
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
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    new_dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
    shuffle_dataset = new_dataset.shuffle(buffer_size=10000)
    repeat_dataset = shuffle_dataset.repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size=batch_size)
    iterator = batch_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    print('NEXT:', next_element['sample'])
    x_split = tf.split(next_element['sample'], num_gpu)
    y_split = tf.split(next_element['label'], num_gpu)
    return x_split, y_split

def _get_input(DATA_PATH=DATA_PATH, BATCH_SIZE = BATCH_SIZE):
    filename_queue = tf.train.string_input_producer([DATA_PATH], shuffle=True, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
          'sample': tf.FixedLenFeature([5], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64)
      })

    samples = tf.cast(features['sample'], tf.float32)
    labels = tf.cast(features['label'], tf.float32)
    samples, labels = tf.train.shuffle_batch([samples, labels],
                                            batch_size=BATCH_SIZE,
                                            capacity=10000,
                                            num_threads=128,
                                            min_after_dequeue=1000)
    # split data based on the number of GPU
    x_split = tf.split(samples, N_GPU)
    y_split = tf.split(labels, N_GPU)
    return x_split, y_split

def _init_parameters():
    w1 = tf.get_variable('w1', shape=[5, 10], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=9))
    b1 = tf.get_variable('b1', shape=[10], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=1))
    w2 = tf.get_variable('w2', shape=[10, 1], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=0))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=2))
    return w1, w2, b1, b2


def _model_nn(w1, w2, b1, b2, x_split, y_split, i_gpu):
    y_hidden = tf.nn.relu(tf.matmul(x_split[i_gpu], w1) + b1)
    y_out = tf.matmul(y_hidden, w2) + b2
    y_out = tf.reshape(y_out, [-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=y_split[i_gpu], name=None)
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = opt.minimize(loss)
    return train

##############    Synchronous data parallelism   ###########################
# initialize must before sess.run(init)
w1, w2, b1, b2 = _init_parameters()
# get train data
x_split, y_split = _get_data()
# set optimizer
opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
#  here is Synchronous Data Parallelism
tower_loss = []
# assign forward propagation to different GPU
for i in range(N_GPU):
    with tf.device("/gpu:%d" % i):
        y_hidden = tf.nn.relu(tf.matmul(x_split[i], w1) + b1)
        y_out = tf.matmul(y_hidden, w2) + b2
        y_out = tf.reshape(y_out, [-1])
        cur_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=y_split[i], name=None)
        tower_loss.append(cur_loss)
    ######  test if parameters are same for different gpu
    ######  must set up a session to obtain detail value of parameters
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(cur_loss)
        print('===============  parameter test sy =========')
        print(i)
        print(sess.run(b1))
        coord.request_stop()
        coord.join(threads)
mean_loss = tf.reduce_mean(tower_loss, 0)

# update parameters。
apply_gradient_op = opt.minimize(mean_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for step in range(2):
        sess.run(apply_gradient_op)
        print('======================  parameter b1  ===================== :')
        print(sess.run(b1))

    coord.request_stop()
    coord.join(threads)
