# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train FER2013 using a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys

import numpy as np
from six.moves import xrange 
import tensorflow as tf

import fer2013


FLAGS = tf.app.flags.FLAGS

local_directory = os.path.dirname(os.path.abspath(__file__))+ '/fer2013/train'

tf.app.flags.DEFINE_string('train_dir', local_directory,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 500000, # actually: 1000000
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('train_batch_size', 128,
                            """Number of images to process in a batch.""")

TRAIN_INPUT_FILE = "Input_Dataset/train.csv"


def train():
    """Train FER-2013 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
    
        # Get images and labels for FER2013.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):  
            images, labels = fer2013.distorted_inputs(TRAIN_INPUT_FILE)
        
        keep_prob = 0.7
    
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = fer2013.inference(images, keep_prob, FLAGS.train_batch_size)


        # Visualize conv1 kernels
        with tf.variable_scope('conv1'):
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights')
            grid = put_kernels_on_grid(weights)
            tf.summary.image('conv1/kernels', grid, max_outputs=1)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Calculate loss.
        loss = fer2013.loss(logits, labels)

        acc = fer2013.accuracy(logits, labels)
    
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = fer2013.train(loss, global_step)
    
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
    
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
    
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
    
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                sess.graph)

        epoch_size = int(fer2013.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.train_batch_size)
        epoch_count = 0

        total_sample_count = epoch_size * FLAGS.train_batch_size

        init_local = tf.local_variables_initializer()
        sess.run(init_local)
    
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if step % 10 == 0:
                accu = sess.run([acc])
                print('Acc: ', accu)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))
            
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step % epoch_size == 0:
                epoch_count += 1
                print('epoch: ' + str(epoch_count))

        print('Training finished')
        true_count = 0
        st = 0
        while st < epoch_size:
            predictions = sess.run([top_k_op])
            true_count += np.sum(predictions)
            st += 1

        accuracy = true_count / total_sample_count
        print("Accuracy: ", accuracy, ' right predicted: ', true_count)


from math import sqrt


def put_kernels_on_grid (kernel, pad = 1):

    """Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.

    Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)

    Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x


def main(argv=None): 
    if tf.gfile.Exists(FLAGS.train_dir):
        print("Folder is not empty, choose another one!")
        sys.exit(0)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
