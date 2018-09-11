# -*- coding: utf-8 -*-

# example to help understand the difference between batch and batch_join and the function of enqueue_many

import tensorflow as tf

tensor_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
print(tensor_list[0][:])

tensor_list2 = [[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]], [[13, 14, 15, 16]], [[17, 18, 19, 20]]]
print(tensor_list2[:][:][0])

with tf.Session() as sess:
    x1 = tf.train.batch(tensor_list, batch_size=2, enqueue_many=False)

    x2 = tf.train.batch(tensor_list, batch_size=2, enqueue_many=True)

    y1 = tf.train.batch_join(tensor_list, batch_size=2, enqueue_many=False)

    y2 = tf.train.batch_join(tensor_list2, batch_size=20, enqueue_many=True)

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("x1 batch:" + "-" * 10)
    print(x1)

    print(sess.run(x1))

    print("x2 batch:" + "-" * 10)
    print(x2)

    print(sess.run(x2))
    print("y1 batch:" + "-" * 10)

    print(sess.run(y1))

    print("y2 batch:" + "-" * 10)

    print(sess.run(y2))
    print("-" * 10)

    coord.request_stop()

    coord.join(threads)

'''
enqueue_many为true时，相当于样本文件中的数据按照某一个维度拼接，其它维度内的数据不断的重复，
[1，2，3，4，1，2，3，4，1...]，[5，6，7，8，5，6，7，8，5...]等等，在同一维度内Batch，这样并不会改变数据的维数。

而batch_join，当enqueue_many为False时，每一个维度看作一个样本，batch_size决定取另一维度的几个，另一维上随机。
当enqueue_many为true时，在另外一个维度上拼接【1，2，3，4 + 9，10，11，12...]拼接的顺序随机，Batch_size决定取多少个


在enqueue_many参数设置为False（默认值）的时候，tf.train.batch的输出，是batch_size * tensor.shape，其含义就是将tensors参数
看做一个样本，那么batch_size个样本，只是单一样本的复制。在其实际应用中，tensors参数一般对应的是一个文件，那么这样操作意味着从文件中
读取batch_size次， 以获得一个batch的样本。

而在enqueu_many参数设置为True的时候，tf.train.batch将tensors参数看做一个batch的样本，那么batch_size只是调整一个batch中样本的维度的，
因为输出的维度是batch_size * tensor.shape[1:]

最后需要注意的tf.train.batch的num_threads参数，指定的是进行入队操作的线程数量，可以进行多线程对于同一个文件进行操作，
这样会比单一线程读取文件快。

tf.train.batch_join一般就对应于多个文件的多线程读取，可以看到当enqueue_many参数设置为False（默认值）的时候，tensor_list中每个tensor
被看做单个样本，这个时候组成batch_size的一个batch的样本，是从各个单一样本中凑成一个batch_size的样本。可以看到由于是多线程，每次取值不同，
也就是类似，每个tensor对应一个文件，也对应一个线程，那么组成batch的时候，该线程读取到文件（例子中是tensor的哪个值）是不确定，这样就形成了
打乱的形成样本的时候。

而在enqueue_many参数设置为True的时候，取一个batch的数据，是在tensor_list中随机取一个，因为每一个就是一个batch的数据，batch_size只是
截断或填充这个batch的大小。
这样就很容易明白了tf.train.batch和tf.train.batch_join的区别，一般来说，单一文件多线程，那么选用tf.train.batch（需要打乱样本，有对应
的tf.train.shuffle_batch）；而对于多线程多文件的情况，一般选用tf.train.batch_join来获取样本（打乱样本同样也有对应的
tf.train.shuffle_batch_join使用）。
'''
