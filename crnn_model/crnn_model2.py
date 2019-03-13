#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-7 下午6:06
# @Author  : YiXin Bao
# @File    : crnn_model.py
"""
Implement the crnn model mentioned in An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition paper
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.contrib import rnn

from crnn_model import cnn_basenet


class ShadowNet(cnn_basenet.CNNBaseModel):
    """
        Implement the crnn model for squence recognition
    """
    def __init__(self, phase: str, hidden_nums: int, layers_nums: int, num_classes: int):
        """

        :param phase: 'Train' or 'Test'
        :param hidden_nums: Number of hidden units in each LSTM cell (block)
        :param layers_nums: Number of LSTM cells (blocks)
        :param num_classes: Number of classes (different symbols) to detect
        """
        super(ShadowNet, self).__init__()
        self.__phase = phase
        self.__hidden_nums = hidden_nums
        self.__layers_nums = layers_nums
        self.__num_classes = num_classes

    @property
    def phase(self):
        """

        :return:
        """
        return self.__phase

    @phase.setter
    def phase(self, value: str):
        """

        :param value:
        :return:
        """
        if not isinstance(value, str) or value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()


    def __input_stem(self, inputdata: tf.Tensor, out_dims: int, name: str=None) -> tf.Tensor:
        """ Resnet-D input stem

        :param inputdata: 4D tensor batch x width x height x channels
        :param out_dims: number of output channels / filters
        :return: the output of the stage
        """
        conv_1 = self.conv2d(inputdata=inputdata, out_channel=32, kernel_size=3, stride=1, use_bias=False, name=name+'_conv1')
        bn_1 = self.layerbn(inputdata=conv_1, is_training=self.phase == 'train')
        relu_1 = self.relu(bn_1)
        conv_2 = self.conv2d(inputdata=relu_1, out_channel=32, kernel_size=3, stride=1, use_bias=False, name=name+'_conv2')
        bn_2 = self.layerbn(inputdata=conv_2, is_training=self.phase == 'train')
        relu_2 = self.relu(bn_2)
        conv_3 = self.conv2d(inputdata=relu_2, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name+'_conv3')
        bn_3 = self.layerbn(inputdata=conv_3, is_training=self.phase == 'train')
        relu_3 = self.relu(bn_3)
        max_pool = self.maxpooling(inputdata=relu_3, kernel_size=2, stride=2)

        return max_pool

    def __bottleneck(self, inputdata: tf.Tensor, out_dims: int, stride: int, downsample: bool, name: str=None):
        identity = inputdata
        conv_1 = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=stride, use_bias=False, name=name+'_conv1')
        bn_1 = self.layerbn(inputdata=conv_1, is_training=self.phase == 'train')
        relu_1 = self.relu(bn_1)
        conv_2 = self.conv2d(inputdata=relu_1, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name+'_conv2')
        out = self.layerbn(inputdata=conv_2, is_training=self.phase == 'train')

        if downsample:
            avg_pool = self.avgpooling(inputdata=identity, kernel_size=stride, stride=stride)
            conv3 = self.conv2d(inputdata=avg_pool, out_channel=out_dims, kernel_size=1, name=name+'_conv3')
            identity = self.layerbn(inputdata=conv3, is_training=self.phase == 'train')

        out = out+identity
        out = self.relu(out)

        return out


    def __feature_sequence_extraction(self, inputdata: tf.Tensor) -> tf.Tensor:
        """ Implements section 2.1 of the paper: "Feature Sequence Extraction"

        :param inputdata: eg. batch*32*100*3 NHWC format
        :return:
        """
        input_stem = self.__input_stem(inputdata=inputdata, out_dims=64, name='input_stem')  # batch*16*50*64
        res1_1 = self.__bottleneck(inputdata=input_stem, out_dims=64, stride=1, downsample=False, name='res1_1') # batch*16*50*64
        res1_2 = self.__bottleneck(inputdata=res1_1, out_dims=64, stride=1, downsample=False, name='res1_2')  # batch*16*50*64
        res2_1 = self.__bottleneck(inputdata=res1_2, out_dims=128, stride=2, downsample=True, name='res2_1') # batch*8*25*128
        res2_2 = self.__bottleneck(inputdata=res2_1, out_dims=128, stride=1, downsample=False, name='res2_2')  # batch*8*25*128
        res3_1 = self.__bottleneck(inputdata=res2_2, out_dims=256, stride=1, downsample=True, name='res3_1')  # batch*8*25*256
        res3_2 = self.__bottleneck(inputdata=res3_1, out_dims=256, stride=1, downsample=False, name='res3_2')  # batch*8*25*256

        max_pool1 = self.maxpooling(inputdata=res3_2, kernel_size=[2, 1], stride=[2, 1], padding='VALID')  # batch*4*25*256
        conv1 = self.conv2d(inputdata=max_pool1, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv1')  # batch*4*25*512
        relu1 = self.relu(conv1)  # batch*4*25*512
        bn1 = self.layerbn(inputdata=relu1, is_training=self.phase == 'train')  # batch*4*25*512
        conv2 = self.conv2d(inputdata=bn1, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv2')  # batch*4*25*512
        bn2 = self.layerbn(inputdata=conv2, is_training=self.phase == 'train')  # batch*4*25*512
        relu2 = self.relu(bn2)  # batch*4*25*512
        max_pool2 = self.maxpooling(inputdata=relu2, kernel_size=[2, 1], stride=[2, 1])  # batch*2*25*512
        conv3 = self.conv2d(inputdata=max_pool2, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv3')  # batch*1*25*512
        bn3 = self.layerbn(inputdata=conv3, is_training=self.phase == 'train')  # batch*1*25*512
        relu3 = self.relu(bn3)  # batch*1*25*512
        return relu3

    def __map_to_sequence(self, inputdata: tf.Tensor) -> tf.Tensor:
        """ Implements the map to sequence part of the network.

        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the lenght of the sequences that the LSTM expects
        :param inputdata:
        :return:
        """
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return self.squeeze(inputdata=inputdata, axis=1)

    def __sequence_label(self, inputdata: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Implements the sequence label part of the network

        :param inputdata:
        :return:
        """
        with tf.variable_scope('LSTMLayers'):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums]*self.__layers_nums]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums]*self.__layers_nums]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,
                                                                         dtype=tf.float32)

            if self.phase.lower() == 'train':
                stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer, keep_prob=0.5)

            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()  # [batch, width, 2*n_hidden]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])  # [batch x width, 2*n_hidden]

            w = tf.Variable(tf.truncated_normal([hidden_nums, self.__num_classes], stddev=0.1), name="w")
            # Doing the affine projection

            logits = tf.matmul(rnn_reshaped, w)

            logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

            # Swap batch and batch axis
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')  # [width, batch, n_classes]

        return rnn_out, raw_pred

    def build_shadownet(self, inputdata: tf.Tensor) -> tf.Tensor:
        """ Main routine to construct the network

        :param inputdata:
        :return:
        """
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)

        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(inputdata=cnn_out)

        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(inputdata=sequence)

        return net_out
