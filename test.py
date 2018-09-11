# import configparser
#
# config = configparser.ConfigParser()
#
# config.read('configTest.txt')
#
# # int
# CNN_LAYER_NUM = config.getint('CNN', 'layer_num')
# print('cnn_layer_num:', CNN_LAYER_NUM)
#
# # array
# CNN_OUTPUT_DIM_FREQ = [int(_) for _ in config.get('CNN', 'output_dim_freq').split(',')]
# print('cnn_output_dim_freq:', CNN_OUTPUT_DIM_FREQ[1])
#
# # string
# CNN_PADDING = [_ for _ in config.get('CNN', 'padding').split(',')]
# print('cnn_padding: ', CNN_PADDING[1])
#
# # Bool
# Linear_svd = config.getboolean('Linear', 'svd')
# print('Linear_svd: ', Linear_svd)


#!/usr/bin/env python
#-*- coding:utf-8 -*-
a = u'🔟'
print(a)
print(type(a))

b = a.encode('utf-8')
print(type(b))
print(b)

d = int.from_bytes(b, 16)
print(d)