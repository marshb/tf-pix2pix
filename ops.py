import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)


def lrelu(x, leaky=0.2):
    return tf.maximum(x, leaky * x)


def conv2x2(x, w):
    return tf.nn.conv2d(x, w, [1, 2, 2, 1], padding='SAME')


def conv1x1(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')


def deconv2x2(x, w):
    input_shape = x.get_shape().as_list()
    out_ch = w.get_shape().as_list()[-2]
    output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, out_ch]
    return tf.nn.conv2d_transpose(x, w, output_shape, [1, 2, 2, 1], padding='SAME')


def sigmoid_xent(logits, targets):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
