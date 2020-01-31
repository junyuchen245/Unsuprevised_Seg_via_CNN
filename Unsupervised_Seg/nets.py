import keras.layers as KL
from keras.layers import *
import sys
from keras.models import Model, load_model
import tensorflow as tf
import keras.constraints as KC
'''
Author: Junyu Chen
Email: jchen245@jhu.edu
'''


def rcnn_block(l, out_num_filters=10, ndims=2, filtersize = 3, trainable_flag=False):
    Conv = getattr(KL, 'Conv%dD' % ndims)
    conv1 = Conv(out_num_filters, filtersize, padding='same')
    stack1 = conv1(l)
    stack2 = BatchNormalization()(stack1)
    stack3 = PReLU()(stack2)
    if not trainable_flag:
        conv1.trainable = False
        stack3.trainable = False
    conv2 = Conv(out_num_filters, filtersize, padding='same', init='he_normal')
    stack4 = conv2(stack3)
    stack5 = Add()([stack1, stack4])
    stack6 = BatchNormalization()(stack5)
    stack7 = PReLU()(stack6)
    if not trainable_flag:
        conv2.trainable = False
        stack7.trainable = False
    conv3 = Conv(out_num_filters, filtersize, padding='same', weights=conv2.get_weights())
    stack8 = conv3(stack7)
    stack9 = Add()([stack1, stack8])
    stack10 = BatchNormalization()(stack9)
    stack11 = PReLU()(stack10)
    if not trainable_flag:
        conv3.trainable = False
        stack11.trainable = False
    conv4 = Conv(out_num_filters, filtersize, padding='same', weights=conv2.get_weights())
    stack12 = conv4(stack11)
    stack13 = Add()([stack1, stack12])
    stack14 = BatchNormalization()(stack13)
    stack15 = PReLU()(stack14)
    if not trainable_flag:
        conv4.trainable = False
        stack15.trainable = False
    return stack15

def recurrentNet_fine_tune(input_size = (384,384,1), ndims=2):
    Conv = getattr(KL, 'Conv%dD' % ndims)

    x_in = Input(input_size)
    x = x_in
    
    for i in range(3): #4
        if i>0:
            x = rcnn_block(x, out_num_filters=16, ndims=2, filtersize = 3, trainable_flag=True)
        else:
            x = rcnn_block(x, out_num_filters=16, ndims=2, filtersize=3, trainable_flag=False)
    # form deformation field
    #x = rcnn_block(x, out_num_filters=16, ndims=2, filtersize=3)
    x = Conv(filters=1, kernel_size=1, padding='same', name='segmentation')(x)
    #x.trainable = False
    x = PReLU()(x)
    #x.trainable = False

    model = Model(inputs=x_in, outputs=x)
    return model


def recurrentNet(input_size=(384, 384, 1), ndims=2):
    Conv = getattr(KL, 'Conv%dD' % ndims)

    x_in = Input(input_size)
    x = x_in

    for i in range(3):  # 4
        x = rcnn_block(x, out_num_filters=16, ndims=2, filtersize=3, trainable_flag=True)
    # form deformation field
    # x = rcnn_block(x, out_num_filters=16, ndims=2, filtersize=3)
    x = Conv(filters=1, kernel_size=1, padding='same', name='segmentation', kernel_constraint=KC.UnitNorm(axis=0))(x)
    x = PReLU()(x)

    model = Model(inputs=x_in, outputs=x)
    return model

