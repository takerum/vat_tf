import time

import numpy
import tensorflow as tf

import layers as L
import vat

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_string('dataset', 'cifar10', "{cifar10, svhn}")

tf.app.flags.DEFINE_string('log_dir', "", "log_dir")
tf.app.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.app.flags.DEFINE_bool('validation', False, "")

tf.app.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 100, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 5, "")
tf.app.flags.DEFINE_integer('num_epochs', 120, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 80, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")

tf.app.flags.DEFINE_string('method', 'vat', "{vat, vatent, baseline}")


if FLAGS.dataset == 'cifar10':
    from cifar10 import inputs, unlabeled_inputs
elif FLAGS.dataset == 'svhn':
    from svhn import inputs, unlabeled_inputs 
else: 
    raise NotImplementedError


NUM_EVAL_EXAMPLES = 5000


def build_training_graph(x, y, ul_x, lr, mom):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    logit = vat.forward(x)
    nll_loss = L.ce_loss(logit, y)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        if FLAGS.method == 'vat':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            additional_loss = vat_loss
        elif FLAGS.method == 'vatent':
            ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
            vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit)
            ent_loss = L.entropy_y_x(ul_logit)
            additional_loss = vat_loss + ent_loss
        elif FLAGS.method == 'baseline':
            additional_loss = 0
        else:
            raise NotImplementedError
        loss = nll_loss + additional_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def build_eval_graph(x, y, ul_x):
    losses = {}
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    at_loss = vat.adversarial_loss(x, y, nll_loss, is_training=True)
    losses['AT_loss'] = at_loss
    ul_logit = vat.forward(ul_x, is_training=False, update_batch_stats=False)
    vat_loss = vat.virtual_adversarial_loss(ul_x, ul_logit, is_training=False)
    losses['VAT_loss'] = vat_loss
    return losses


def main(_):
    print(FLAGS.epsilon, FLAGS.top_bn)
    numpy.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(numpy.random.randint(1234))
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):
            images, labels = inputs(batch_size=FLAGS.batch_size,
                                    train=True,
                                    validation=FLAGS.validation,
                                    shuffle=True)
            ul_images = unlabeled_inputs(batch_size=FLAGS.ul_batch_size,
                                         validation=FLAGS.validation,
                                         shuffle=True)

            images_eval_train, labels_eval_train = inputs(batch_size=FLAGS.eval_batch_size,
                                                          train=True,
                                                          validation=FLAGS.validation,
                                                          shuffle=True)
            ul_images_eval_train = unlabeled_inputs(batch_size=FLAGS.eval_batch_size,
                                                    validation=FLAGS.validation,
                                                    shuffle=True)

            images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
                                                        train=False,
                                                        validation=FLAGS.validation,
                                                        shuffle=True)

        with tf.device(FLAGS.device):
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")
            with tf.variable_scope("CNN") as scope:
                # Build training graph
                loss, train_op, global_step = build_training_graph(images, labels, ul_images, lr, mom)
                scope.reuse_variables()
                # Build eval graph
                losses_eval_train = build_eval_graph(images_eval_train, labels_eval_train, ul_images_eval_train)
                losses_eval_test = build_eval_graph(images_eval_test, labels_eval_test, images_eval_test)

            init_op = tf.global_variables_initializer()

        if not FLAGS.log_dir:
            logdir = None
            writer_train = None
            writer_test = None
        else:
            logdir = FLAGS.log_dir
            writer_train = tf.summary.FileWriter(FLAGS.log_dir + "/train", g)
            writer_test = tf.summary.FileWriter(FLAGS.log_dir + "/test", g)

        saver = tf.train.Saver(tf.global_variables())
        sv = tf.train.Supervisor(
            is_chief=True,
            logdir=logdir,
            init_op=init_op,
            init_feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1},
            saver=saver,
            global_step=global_step,
            summary_op=None,
            summary_writer=None,
            save_model_secs=150, recovery_wait_secs=0)

        print("Training...")
        with sv.managed_session() as sess:
            for ep in range(FLAGS.num_epochs):
                if sv.should_stop():
                    break

                if ep < FLAGS.epoch_decay_start:
                    feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                else:
                    decayed_lr = ((FLAGS.num_epochs - ep) / float(
                        FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                    feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}

                sum_loss = 0
                start = time.time()
                for i in range(FLAGS.num_iter_per_epoch):
                    _, batch_loss, _ = sess.run([train_op, loss, global_step],
                                                feed_dict=feed_dict)
                    sum_loss += batch_loss
                end = time.time()
                print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch, "elapsed_time:", end - start)

                if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs:
                    # Eval on training data
                    act_values_dict = {}
                    for key, _ in losses_eval_train.iteritems():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES / FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        values = losses_eval_train.values()
                        act_values = sess.run(values)
                        for key, value in zip(act_values_dict.keys(), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.iteritems():
                        print("train-" + key, value / n_iter_per_epoch)
                        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                    if writer_train is not None:
                        writer_train.add_summary(summary, current_global_step)

                    # Eval on test data
                    act_values_dict = {}
                    for key, _ in losses_eval_test.iteritems():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES / FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        values = losses_eval_test.values()
                        act_values = sess.run(values)
                        for key, value in zip(act_values_dict.keys(), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.iteritems():
                        print("test-" + key, value / n_iter_per_epoch)
                        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                    if writer_test is not None:
                        writer_test.add_summary(summary, current_global_step)

            saver.save(sess, sv.save_path, global_step=global_step)
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
