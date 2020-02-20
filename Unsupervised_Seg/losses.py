import tensorflow as tf
from keras import backend as K
import numpy as np
'''
Active Contour Without Edge loss
Author: Junyu Chen
'''

'''
Unsupervised loss
'''
class ACWE():
    def __init__(self, maxVal=1.0, areaWt = 0.004):
        self.maxVal = maxVal
        self.areaWt = areaWt

    def acwe(self, y_true, y_pred):

        y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=1)
        mean_in = tf.reduce_sum(tf.multiply(y_pred, y_true)) / tf.cast(tf.math.count_nonzero(y_pred), tf.float32)
        mean_out = tf.reduce_sum(tf.multiply(1 - y_pred, y_true)) / tf.cast(tf.math.count_nonzero(1 - y_pred), tf.float32)
        E_in = tf.reduce_sum(tf.multiply(y_pred, K.pow((y_true - mean_in), 2)))
        E_out = tf.reduce_sum(tf.multiply(1 - y_pred, K.pow((y_true - mean_out), 2)))
        E_area = tf.reduce_sum(y_pred)#tf.cast(tf.math.count_nonzero(y_pred), tf.float32))
        return tf.abs(E_in+E_out+self.areaWt*E_area)

    def loss(self, I, J):
        return self.acwe(I,J)

'''
label loss:
modified from https://github.com/xuuuuuuchen/Active-Contour-Loss
Xu  Chen,  Bryan  M  Williams,  Srinivasa  R  Vallabhaneni,  Gabriela  Czanner,  and  RachelWilliams.  
Learning Active Contour Models for Medical Image Segmentation.
Cvpr2019, pages 11632–11640, 2019
'''
class ACWE_label():
    def __init__(self, maxVal=1.0, areaWt = 0.004):
        self.maxVal = maxVal
        self.areaWt = areaWt

    def Active_Contour_Loss(self, y_true, y_pred):
        # y_pred = K.cast(y_pred, dtype = 'float64')

        """
        lenth term
        """
        #_0 = tf.constant([0.0])
        #y_pred = tf.cast(tf.math.greater_equal(y_pred, _0), tf.float32)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=1)
        x = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]  # horizontal and vertical directions
        y = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

        delta_x = x[:, 1:, :-2, :] ** 2
        delta_y = y[:, :-2, 1:, :] ** 2
        delta_u = K.abs(delta_x + delta_y)

        epsilon = 0.00000001  # where is a parameter to avoid square root is zero in practice.
        w = 1
        lenth = w * K.sum(K.sqrt(delta_u + epsilon))  # equ.(11) in the paper

        """
        region term
        """

        C_1 = np.ones((192, 192))
        C_2 = np.zeros((192, 192))

        region_in = K.abs(K.sum(y_pred * ((y_true - 1) ** 2)))  # equ.(12) in the paper
        region_out = K.abs(K.sum((1 - y_pred) * ((y_true - 0) ** 2)))  # equ.(12) in the paper

        lambdaP = 1  # lambda parameter could be various.

        loss = lenth + lambdaP * (region_in + region_out)

        return loss

    def loss(self, I, J):
        return self.Active_Contour_Loss(I,J)
