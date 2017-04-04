import numpy as np
import tensorflow as tf

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def conv_bn_relu(inpt, filter_shape, stride):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out

def conv_relu(inpt, filter_shape, stride):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

    out = tf.nn.relu(conv)
    
    return out

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    out = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    
    return out

def residual_block(inpt, output_depth, projection=False):
    input_depth = inpt.get_shape().as_list()[3]

    conv1 = conv_bn_relu(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_bn_relu(conv1, [3, 3, output_depth, output_depth], 1)
    
    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer

    return res
