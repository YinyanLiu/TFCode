# import configparser
#
# config = configparser.ConfigParser()
#
# config.read('CRnn_configure.txt')
#
# # int
# LAYER_NUMs = config.get('Layer5', 'layer_name')
# print('cnn_layer_num:', LAYER_NUMs)
#
# # array
# OUTPUT_DIM = [int(_) for _ in config.get('Layer0','input_dim').split(',')][0]
# print(OUTPUT_DIM)
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

import fst


