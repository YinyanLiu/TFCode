# -*- coding: UTF-8 -*-
# !/usr/bin/python3
# Env: python3.6
import tensorflow as tf
import numpy as np
import os

data_filename1 = 'data/data_train1.txt'  # 生成txt数据保存路径
data_filename2 = 'data/data_train2.txt'  # 生成txt数据保存路径
tfrecord_path1 = 'data/test_data1.tfrecord'  # tfrecord1文件保存路径
tfrecord_path2 = 'data/test_data2.tfrecord'  # tfrecord2文件保存路径

##############################  读取txt文件，并转为tfrecord文件 ###########################
# every line of data is just as follow: 1 2 3 4 5/1. train data: 1 2 3 4 5, label: 1
def txt_to_tfrecord(txt_filename, tfrecord_path):
    # 第一步：生成TFRecord Writer
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    # 第二步：读取TXT数据，并分割出样本数据和标签
    file = open(txt_filename)
    for data_line in file.readlines():  # 每一行
        data_line = data_line.strip('\n')  # 去掉换行符
        sample = []
        spls = data_line.split('/', 1)[0]  # 样本
        for m in spls.split(' '):
            sample.append(int(m))
        label = data_line.split('/', 1)[1]  # 标签
        label = int(label)

        # 第三步： 建立feature字典，tf.train.Feature()对单一数据编码成feature
        feature = {'sample': tf.train.Feature(int64_list=tf.train.Int64List(value=sample)),
                   'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}
        # 第四步：可以理解为将内层多个feature的字典数据再编码，集成为features
        features = tf.train.Features(feature=feature)
        # 第五步：将features数据封装成特定的协议格式
        example = tf.train.Example(features=features)
        # 第六步：将example数据序列化为字符串
        Serialized = example.SerializeToString()
        # 第七步：将序列化的字符串数据写入协议缓冲区
        writer.write(Serialized)
    # 记得关闭writer和open file的操作
    writer.close()
    file.close()
    return
txt_to_tfrecord(txt_filename=data_filename1, tfrecord_path=tfrecord_path1)
txt_to_tfrecord(txt_filename=data_filename2, tfrecord_path=tfrecord_path2)


# 第一步： 建立文件名队列
filename_queue = tf.train.string_input_producer([tfrecord_path1, tfrecord_path2])
def read_single(filename_queue, shuffle_batch, if_enq_many):
    # 第二步： 建立阅读器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 第三步：根据写入时的格式建立相对应的读取features
    features = {
        'sample': tf.FixedLenFeature([5], tf.int64),  # 如果不是标量，一定要在这里说明数组的长度
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # 第四步： 用tf.parse_single_example()解析单个EXAMPLE PROTO
    Features = tf.parse_single_example(serialized_example, features)

    # 第五步：对数据进行后处理
    sample = tf.cast(Features['sample'], tf.float32)
    label = tf.cast(Features['label'], tf.float32)

    # 第六步：生成Batch数据 generate batch
    if shuffle_batch:  # 打乱数据顺序，随机取样
        sample_single, label_single = tf.train.shuffle_batch([sample, label],
                                                             batch_size=1,
                                                             capacity=10000,
                                                             min_after_dequeue=1000,
                                                             num_threads=1,
                                                             enqueue_many=if_enq_many)  # 主要是为了评估enqueue_many的作用
    else:  # # 如果不打乱顺序则用tf.train.batch(), 输出队列按顺序组成Batch输出

        ###################### multi reader, multi samples, please code as below     ###############################
        '''
        example_list = [[sample,label] for _ in range(2)]  # Reader设置为2

        sample_single, label_single = tf.train.batch_join(example_list, batch_size=3)
        '''
        #######################  single reader, single sample,  please set batch_size = 1   #########################
        #######################  single reader, multi samples,  please set batch_size = batch_size    ###############
        sample_single, label_single = tf.train.batch([sample, label],
                                                     batch_size=1,
                                                     capacity=10000,
                                                     num_threads=1,
                                                     enqueue_many=if_enq_many)

    return sample_single, label_single

x1_samples, y1_labels = read_single(filename_queue, shuffle_batch=False, if_enq_many=False)

# 定义初始化变量范围
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # 初始化
    # 如果tf.train.string_input_producer([tfrecord_path], num_epochs=30)中num_epochs不为空的化，必须要初始化local变量
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()  # 管理线程
    threads = tf.train.start_queue_runners(coord=coord)  # 文件名开始进入文件名队列和内存
    for i in range(5):
        # Queue + tf.parse_single_example()读取tfrecord文件
        X1, Y1 = sess.run([x1_samples, y1_labels])
        print('X1: ', X1, 'Y1: ', Y1)
        # Queue + tf.parse_example()读取tfrecord文件

    coord.request_stop()
    coord.join(threads)
