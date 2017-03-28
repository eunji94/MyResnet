import tensorflow as tf
from resnet import conv_layer_relu, conv_layer_bn_relu, conv_layer, residual_block

def resnet(inpt, n):
    if n < 20 or (n - 20) % 12 != 0:
        print "ResNet depth invalid."
        return

    num_conv = (n - 20) / 4 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer_relu(inpt, [3, 3, 3, 64], 1)
        layers.append(conv1)

    for i in range (num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 64)
            conv2 = residual_block(conv2_x, 64)
            layers.append(conv2_x)
            layers.append(conv2)

        out = conv_layer(layers[-1], [3, 3, 64, 3], 1)
        layers.append(out)

    return layers[-1]
