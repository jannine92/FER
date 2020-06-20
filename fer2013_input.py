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


"""Routine for decoding the FER2013 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf




IMAGE_SIZE = 32

# Global constants describing the data set.
NUM_CLASSES = 7
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 28709 #
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3589 
NUM_EPOCHS_TEST = 1


def read_fer2013(filename_queue, make_prediction):
    """Reads and parses examples from FER2013 data files.
    
    Args:
      filename_queue: A queue of strings with the filenames to read from.
      make_prediction: bool to determine whether you are training / testing or making a prediction
    
    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (48)
        width: number of columns in the result (48)
        depth: number of color channels in the result (1)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class FER2013Record(object):
        pass # placeholder for an empty block because you want to return it as object
    result = FER2013Record()
    
    result.height = 48
    result.width = 48
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    
    
    readerCSV = tf.TextLineReader()
   
    # key: file, record, value: string value
    result.key, value = readerCSV.read(filename_queue)
   
    # default values, in case of empty columns
    # specifies also type of the decoded result
    #record_defaults = [[1], [1]]
    if(make_prediction):
        r_defaults = [[1] for x in range(image_bytes)]
    else:
        r_defaults = [[1] for x in range(image_bytes + 1)]
        
    values = tf.decode_csv(value, record_defaults=r_defaults)
   
    # structure make_prediction input: 48x48
    if(not make_prediction):
        result.label, image = tf.split(values, [1, 2304], 0)
        result.label = tf.cast(result.label, tf.int32)
    else:
        result.label = tf.cast(-1, tf.int32)
        image = values   
    
    image = tf.cast(image, tf.uint8)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(image, [result.depth, result.height, result.width])
    
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
                
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    
    Args:
      image: 3-D Tensor of [height, width, 1] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: shuffle examples
    
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)


    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size, train_input_file):
    """Construct distorted input for training using the Reader ops.
    
    Args:
      data_dir: Path to the data directory.
      batch_size: Number of images per batch.
      train_input_file: input file for training
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """

    filename = [os.path.join(data_dir, train_input_file)]
    
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filename)
    
    read_input = read_fer2013(filename_queue, False)
    
    
    # Read examples from files in the filename queue.
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 1])
    
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)
    
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 1])
    read_input.label.set_shape([1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d FER2013 images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size, input_file):
    """Construct input for evaluation using the Reader ops.
    
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the FER-2013 data directory.
      batch_size: Number of images per batch.
      input_file: input file with data
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    
        
    if (eval_data == 'train'): 
        filename = [os.path.join(data_dir, input_file)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    elif (eval_data == 'test'):
        filename = [os.path.join(data_dir, input_file)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    else: # eval_data = 'make_prediction'
        filename = [os.path.join(data_dir, input_file)] # read image to make_prediction
        num_examples_per_epoch = 1
    

    # Create a queue that produces the filenames to read.
    if(eval_data == 'test'):
        filename_queue = tf.train.string_input_producer(filename)
    else:
        filename_queue = tf.train.string_input_producer(filename)
    
    read_input = read_fer2013(filename_queue, eval_data == 'make_prediction')
    
    # Read examples from files in the filename queue.
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)
    
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)
    
    # Set the shapes of tensors.
    #float_image.set_shape([height, width, 1])
    if(not eval_data == 'make_prediction'):
        read_input.label.set_shape([1])
    
    if(eval_data == 'make_prediction'):
        image = tf.reshape(float_image, (1, 32, 32, 1))
        return image, read_input.label
    else:
        # --------------- until here necessary for make_prediction as well
        
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        
        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(float_image, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=False)
