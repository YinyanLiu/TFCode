# -*- coding: UTF-8 -*-
# !/usr/bin/python3
# Env: python3.6
import tensorflow as tf
import numpy as np
import os

# path
data_filename = 'data/data_train.txt'  # 生成txt数据保存路径
size = (10000, 5)
tfrecord_path = 'data/test_data.tfrecord'  # tfrecord文件保存路径

########################### 生成txt数据 10000个样本。########################
def generate_data(data_filename=data_filename, size=size):
    if not os.path.exists(data_filename):
        np.random.seed(9)
        x_data = np.random.randint(0, 10, size=size)
        y1_data = np.ones((size[0] // 2, 1), int)  # 一半标签是0，一半是1
        y2_data = np.zeros((size[0] // 2, 1), int)
        y_data = np.append(y1_data, y2_data)
        np.random.shuffle(y_data)

        xy_data = str('')
        for xy_row in range(len(x_data)):
            x_str = str('')
            for xy_col in range(len(x_data[0])):
                if not xy_col == (len(x_data[0]) - 1):
                    x_str = x_str + str(x_data[xy_row, xy_col]) + ' '
                else:
                    x_str = x_str + str(x_data[xy_row, xy_col])
            y_str = str(y_data[xy_row])
            xy_data = xy_data + (x_str + '/' + y_str + '\n')

        # write to txt
        write_txt = open(data_filename, 'w')
        write_txt.write(xy_data)
        write_txt.close()
    return

##############################  读取txt文件，并转为tfrecord文件 ###########################
# every line of data is just as follow: 1 2 3 4 5/1. train data: 1 2 3 4 5, label: 1
def txt_to_tfrecord(txt_filename=data_filename, tfrecord_path=tfrecord_path):
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
        print('sample:', sample, 'labels:', label)

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


##########################   用Queue方式中的tf.parse_single_example解析tfrecord  #########################

# 第一步： 建立文件名队列
filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=30, shuffle=True)


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
                                                             batch_size=2,
                                                             capacity=10000,
                                                             min_after_dequeue=1000,
                                                             num_threads=1,
                                                             enqueue_many=if_enq_many)  # 主要是为了评估enqueue_many的作用
    else:  # # 如果不打乱顺序则用tf.train.batch(), 输出队列按顺序组成Batch输出
        '''
        example_list = [[sample,label] for _ in range(2)]  # Reader设置为2

        sample_single, label_single = tf.train.batch_join(example_list, batch_size=1)
        '''

        sample_single, label_single = tf.train.batch([sample, label],
                                                     batch_size=2,
                                                     capacity=10000,
                                                     num_threads=1,
                                                     enqueue_many=if_enq_many)

    return sample_single, label_single


###############################   用Queue方式中的tf.parse_example解析tfrecord  ##################################

def read_parse(filename_queue, shuffle_batch, if_enq_many):
    # 第二步： 建立阅读器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 第三步： 设置shuffle_batch
    if shuffle_batch:
        batch = tf.train.shuffle_batch([serialized_example],
                                       batch_size=3,
                                       capacity=10000,
                                       min_after_dequeue=1000,
                                       num_threads=1,
                                       enqueue_many=if_enq_many)  # 主要是为了评估enqueue_many的作用

    else:
        batch = tf.train.batch([serialized_example],
                               batch_size=3,
                               capacity=10000,
                               num_threads=1,
                               enqueue_many=if_enq_many)
        # 第四步：根据写入时的格式建立相对应的读取features
    features = {
        'sample': tf.FixedLenFeature([5], tf.int64),  # 如果不是标量，一定要在这里说明数组的长度
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # 第五步： 用tf.parse_example()解析多个EXAMPLE PROTO
    Features = tf.parse_example(batch, features)

    # 第六步：对数据进行后处理
    samples_parse = tf.cast(Features['sample'], tf.float32)
    labels_parse = tf.cast(Features['label'], tf.float32)
    return samples_parse, labels_parse


###################################### 用Dataset读取tfrecord文件  ###############################################

# 定义解析函数
def _parse_function(example_proto):
    dics = {  # 这里没用default_value，随后的都是None
        'sample': tf.FixedLenFeature([5], tf.int64),  # 如果不是标量，一定要在这里说明数组的长度
        'label': tf.FixedLenFeature([], tf.int64)}
    # 把序列化样本和解析字典送入函数里得到解析的样本
    parsed_example = tf.parse_single_example(example_proto, dics)

    parsed_example['sample'] = tf.cast(parsed_example['sample'], tf.float32)
    parsed_example['label'] = tf.cast(parsed_example['label'], tf.float32)
    # 返回所有feature
    return parsed_example


def read_dataset(tfrecord_path=tfrecord_path):
    # 声明阅读器
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    # 建立解析函数，其中num_parallel_calls指定并行线程数
    new_dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_function, batch_size=2))
    # 打乱样本顺序
    shuffle_dataset = new_dataset.shuffle(buffer_size=20000)
    new_dataset = new_dataset.repeat(10)  # 设置epoch次数为10
    # 数据提前进入队列
    prefetch_dataset = new_dataset.prefetch(2000)
    # 建立迭代器
    iterator = prefetch_dataset.make_one_shot_iterator()
    # 获得下一个样本
    next_element = iterator.get_next()
    return next_element


################################################## 建立graph ####################################

# 生成数据
# generate_data()
# 读取数据转为tfrecord文件
# txt_to_tfrecord()
# Queue + tf.parse_single_example()读取tfrecord文件
x1_samples, y1_labels = read_single(filename_queue, shuffle_batch=False, if_enq_many=False)
# Queue + tf.parse_example()读取tfrecord文件
x2_samples, y2_labels = read_parse(filename_queue, shuffle_batch=True, if_enq_many=False)
# Dataset读取数据
next_element = read_dataset()

# 定义初始化变量范围
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # 初始化
    # 如果tf.train.string_input_producer([tfrecord_path], num_epochs=30)中num_epochs不为空的化，必须要初始化local变量
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()  # 管理线程
    threads = tf.train.start_queue_runners(coord=coord)  # 文件名开始进入文件名队列和内存
    for i in range(1):
        # Queue + tf.parse_single_example()读取tfrecord文件
        X1, Y1 = sess.run([x1_samples, y1_labels])
        print('X1: ', X1, 'Y1: ', Y1)
        # Queue + tf.parse_example()读取tfrecord文件
        X2, Y2 = sess.run([x2_samples, y2_labels])
        print('X2: ', X2, 'Y2: ', Y2)
        # Dataset读取数据
        print('dataset:', sess.run([next_element['sample'],
                                    next_element['label']]))

    coord.request_stop()
    coord.join(threads)
