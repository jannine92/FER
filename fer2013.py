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

"""Builds the network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

import tensorflow as tf

import fer2013_input


FLAGS = tf.app.flags.FLAGS

local_directory = os.path.dirname(os.path.abspath(__file__))+ '/fer2013' + '/'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', local_directory,
                           """Path to the fer-2013 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")


# Global constants describing the fer2013 data set.
IMAGE_SIZE = fer2013_input.IMAGE_SIZE
NUM_CLASSES = fer2013_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = fer2013_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = fer2013_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-13     # Initial learning rate. # initial: 0.1, then: 1e-08, 1e-10

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

      Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    
      Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

      Note that the Variable is initialized with a truncated normal distribution.
      A weight decay is added only if one is specified.
    
      Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    
      Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs(train_input_file):
    """Construct distorted input for training using the Reader ops.
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    print('distorted inputs data dir', data_dir)
    
    images, labels = fer2013_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size, 
                                        train_input_file=train_input_file)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels



def inputs(eval_data, input_file):
    """Construct input for evaluation using the Reader ops.
    
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      input_file: input file with data
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images, labels = fer2013_input.inputs(eval_data=eval_data, data_dir=data_dir,
                                batch_size=FLAGS.batch_size,
                                input_file=input_file)
    
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images, keep_prob, batch_size):
    """Build the FER2013 model.
    
    Args:
      images: Images returned from distorted_inputs() or inputs().
      keep_prob: keep probability for Dropout layer
      batch_size: batch size
    
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    epsilon = 1e-3

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 1, 128],
                                             stddev=1e-4, wd=0.0) 
        # conv2d(input, filter, strides, padding, use_cudnn_on_gpu)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') #use_cudnn_on_gpu=True is default
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

        batch_mean_1, batch_var_1 = tf.nn.moments(conv1, [0])
        scale_1 = _variable_on_cpu('scales', [128], tf.constant_initializer(1.0))
        beta_1 = _variable_on_cpu('betas', [128], tf.constant_initializer(0.0))

        batch_normalization_1 = tf.nn.batch_normalization(conv1, batch_mean_1, batch_var_1, beta_1, scale_1, epsilon)

        _activation_summary(conv1)


    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1 (local response normalization -> form of lateral inhibition )
    # -> excited neurons reduce activity of neighbors so that only the output of the
    # strongest is fired
    '''norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')'''

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

        batch_mean_2, batch_var_2 = tf.nn.moments(conv2, [0])
        scale_2 = _variable_on_cpu('scales', [256], tf.constant_initializer(1.0))
        beta_2 = _variable_on_cpu('betas', [256], tf.constant_initializer(0.0))

        batch_normalization_2 = tf.nn.batch_normalization(conv2, batch_mean_2, batch_var_2, beta_2, scale_2, epsilon)

        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    dropout_c2 = tf.nn.dropout(pool2, keep_prob)

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(dropout_c2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)

        batch_mean_3, batch_var_3 = tf.nn.moments(conv3, [0])
        scale_3 = _variable_on_cpu('scales', [256], tf.constant_initializer(1.0))
        beta_3 = _variable_on_cpu('betas', [256], tf.constant_initializer(0.0))

        batch_normalization_3 = tf.nn.batch_normalization(conv3, batch_mean_3, batch_var_3, beta_3, scale_3, epsilon)

        _activation_summary(conv3)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    dropout_c3 = tf.nn.dropout(pool3, keep_prob)

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(dropout_c3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)

        batch_mean_4, batch_var_4 = tf.nn.moments(conv4, [0])
        scale_4= _variable_on_cpu('scales', [256], tf.constant_initializer(1.0))
        beta_4 = _variable_on_cpu('betas', [256], tf.constant_initializer(0.0))

        batch_normalization_4 = tf.nn.batch_normalization(conv4, batch_mean_4, batch_var_4, beta_4, scale_4, epsilon)

        _activation_summary(conv4)

        # norm4

    # pool4
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    dropout_c4 = tf.nn.dropout(pool4, keep_prob)

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(dropout_c4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)

        batch_mean_5, batch_var_5 = tf.nn.moments(conv5, [0])
        scale_5 = _variable_on_cpu('scales', [256], tf.constant_initializer(1.0))
        beta_5 = _variable_on_cpu('betas', [256], tf.constant_initializer(0.0))

        batch_normalization_5 = tf.nn.batch_normalization(conv5, batch_mean_5, batch_var_5, beta_5, scale_5, epsilon)

        _activation_summary(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    dropout_c5 = tf.nn.dropout(pool5, keep_prob)

    # local4
    with tf.variable_scope('local4') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in dropout_c5.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(dropout_c5, [batch_size, dim])
        
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local4)
        dropout1 = tf.nn.dropout(local4, keep_prob)

    # local5
    with tf.variable_scope('local5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(dropout1, weights) + biases, name=scope.name)
        _activation_summary(local5)
        # keep_prob = 0.5 # only during training, else: 1.0
        dropout2 = tf.nn.dropout(local5, keep_prob)
    
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(dropout2, weights), biases, name=scope.name) # dropout instead of local5
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    
    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def accuracy(logits, labels):

    labels = tf.cast(labels, tf.int64)
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.add_to_collection('accuracy', accuracy)
    tf.summary.scalar('Training Accuracy', accuracy)
    return tf.get_collection('accuracy')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in FER2013 model.
    
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train FER2013 model.
    
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size #224 batches
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)
    
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        # tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        # beta1: exponential decay rate for the 1st moment estimates
        # beta2: exponential decay rate for the 2nd moment estimates
        # epsilon: small constant for numerical stability
        # use_locking: if True use locks for update operations
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    return train_op

