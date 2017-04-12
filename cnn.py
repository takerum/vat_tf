import tensorflow as tf
import numpy
import sys, os
import layers as L

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.app.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.app.flags.DEFINE_boolean('top_bn', False, "")


def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    h = x

    rng = numpy.random.RandomState(seed)

    h = L.conv(h, ksize=3, stride=1, f_in=3, f_out=128, seed=rng.randint(123456), name='c1')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b1'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c2')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b2'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c3')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b3'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)
    h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h

    h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=256, seed=rng.randint(123456), name='c4')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b4'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=256, seed=rng.randint(123456), name='c5')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b5'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=256, seed=rng.randint(123456), name='c6')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b6'), FLAGS.lrelu_a)

    h = L.max_pool(h, ksize=2, stride=2)
    h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h

    h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=512, seed=rng.randint(123456), padding="VALID", name='c7')
    h = L.lrelu(L.bn(h, 512, is_training=is_training, update_batch_stats=update_batch_stats, name='b7'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=1, stride=1, f_in=512, f_out=256, seed=rng.randint(123456), name='c8')
    h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b8'), FLAGS.lrelu_a)
    h = L.conv(h, ksize=1, stride=1, f_in=256, f_out=128, seed=rng.randint(123456), name='c9')
    h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b9'), FLAGS.lrelu_a)

    h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
    h = L.fc(h, 128, 10, seed=rng.randint(123456), name='fc')

    if FLAGS.top_bn:
        h = L.bn(h, 10, is_training=is_training,
                 update_batch_stats=update_batch_stats, name='bfc')

    return h
