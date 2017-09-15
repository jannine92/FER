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


"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.

IMAGE_SIZE = 32
#IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 7
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 28709 # training set 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3589 # public test set 10000
NUM_EPOCHS_TEST = 1
#TRAIN_INPUT_FILE = "train.csv"
#TEST_INPUT_FILE = "test.csv"


"""
def read_fer2013(filename):
    Reads and parses examples from CIFAR10 data files.
    
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    
    Args:
      filename_queue: A queue of strings with the filenames to read from.
    
    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
   
    class FER2013Record(object):
        pass
    result = FER2013Record()
    
    label_bytes = 1
    result.height = 48
    result.width = 48
    result.depth = 1 # 3 for RGB
    
    image_bytes = result.height * result.width * result.depth
    
    with open(filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = datareader.next()
        print(headers)
        
        for row in datareader:
            emotion = row[0]
            pixels = map(tf.uint8, row[1].split()) #or: int instead of uint8
            usage = row[2]
            pixel_array = np.asarray(pixels)
            
            
            image = pixel_array.reshape(result.height, result.width)
            
        
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    #depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
     #                        [result.depth, result.height, result.width])
    
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    #label_bytes = 1  # 2 for CIFAR-100
    #===========================================================================
    # result.height = 32 # 48
    # result.width = 32 # 48
    # result.depth = 3
    # image_bytes = result.height * result.width * result.depth
    #===========================================================================
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return result
"""

def read_fer2013(filename_queue, make_prediction):
    """Reads and parses examples from FER2013 data files.
    
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    
    Args:
      filename_queue: A queue of strings with the filenames to read from.
    
    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class FER2013Record(object):
        pass # placeholder for an empty block because you want to return it as object
    result = FER2013Record()
    
    #label_bytes = 1 
    result.height = 48 # 32
    result.width = 48 # 32
    result.depth = 1 # 3
    image_bytes = result.height * result.width * result.depth
    
    
    readerCSV = tf.TextLineReader()
    
   
    
    # key: file, record, value: string value
    result.key, value = readerCSV.read(filename_queue)
    #print(tf.shape(value))
    

#tensorflow.python.framework.errors_impl.InvalidArgumentError: Expect 2304 fields but have 48 in record 0

        
    # default values, in case of empty columns
    # specifies also type of the decoded result
    #record_defaults = [[1], [1]]
    if(make_prediction):
        r_defaults = [[1] for x in range(image_bytes)]
    else:
        r_defaults = [[1] for x in range(image_bytes + 1)]
        
    values = tf.decode_csv(value, record_defaults=r_defaults)
    #print(label_value.get_shape())
    #image_converted = tf.as_string(image_values)
    #print(tf.shape(row_values))
    #print(row_values)
    
    #features = tf.stack(image_values)
    # print(tf.shape(features))

    #images = tf.string_split([image_converted])
    #print("Images: ", tf.shape(images))
   
    # structure make_prediction input: 48x48
    if(not make_prediction):
        result.label, image = tf.split(values, [1, 2304], 0)
        result.label = tf.cast(result.label, tf.int32)
    else:
        result.label = tf.cast(-1, tf.int32)
        image = values   
    # The first bytes represent the label, which we convert from uint8->int32.
    #result.label = tf.cast(
        #tf.slice(values, [0], [label_bytes]), tf.int32)
    
    image = tf.cast(image, tf.uint8)
    
    #print("result label: ")
    #print(tf.shape(result.label))
    #print("result image: ")
    #print(tf.shape(image))


    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(image, [result.depth, result.height, result.width])
    
    
    #print(tf.shape(depth_major))
    
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
                
    

    # ---------------------------- Old version with binary files: ----------------------------
    
    # Dimensions of the images  dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    """label_bytes = 1  # 2 for CIFAR-100
    result.height = 48 # 32
    result.width = 48 # 32
    result.depth = 1 # 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    
    result.key, value = reader.read(filename_queue)
        
    
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8) 
    # decode_raw returns: A `Tensor` of type `out_type`.
    # A Tensor with one more dimension than the input `bytes`.  The
    # added dimension will have size equal to the length of the elements
    # of `bytes` divided by the number of bytes to represent `out_type`.
    
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])"""
    
    return result


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.
    
    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.
    
    Args:
      filename_queue: A queue of strings with the filenames to read from.
    
    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass # placeholder for an empty block because you want to return it as object
    result = CIFAR10Record()
    
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32 # 48
    result.width = 32 # 48
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
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
    """Construct distorted input for CIFAR training using the Reader ops.
    
    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
     """   
    
    
    #filename = [os.path.join(data_dir, 'train.bin')]
    filename = [os.path.join(data_dir, train_input_file)]
    
    #if not tf.gfile.Exists(filename):
    #   raise ValueError('Failed to find file: ' + filename)

   
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filename)
    
    
    
    read_input = read_fer2013(filename_queue, False)
    
    
    # Read examples from files in the filename queue.
    #read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 1]) # 3
    
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
    """Construct input for CIFAR evaluation using the Reader ops.
    
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the FER-2013 data directory.
      batch_size: Number of images per batch.
    
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """

    """filename = [os.path.join(data_dir, TEST_INPUT_FILE)]
    #filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL"""
    
        
    if (eval_data == 'train'): # = train
        filename = [os.path.join(data_dir, input_file)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    elif (eval_data == 'test'):
        filename = [os.path.join(data_dir, input_file)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    else: # eval_data = 'make_prediction'
        filename = [os.path.join(data_dir, input_file)] # read image to make_prediction
        num_examples_per_epoch = 1
        # anderer Ablauf: nur eine Datei einlesen -> need of return function?
        # TODO: write own method for read_fer2013 because file format is different: label is missing
        # --> queue eigentlich nicht noetig, also '_generate_image_and_label_batch'-Methode
    
    #if not tf.gfile.Exists(filename):
    #       raise ValueError('Failed to find file: ' + filename)

    # Create a queue that produces the filenames to read.
    if(eval_data == 'test'):
        filename_queue = tf.train.string_input_producer(filename) # TODO: change back to 1
    else:
        filename_queue = tf.train.string_input_producer(filename)
    
    read_input = read_fer2013(filename_queue, eval_data == 'make_prediction')
    
    # Read examples from files in the filename queue.
    # read_input = read_cifar10(filename_queue)
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
        # --------------- bis hier auch noetig fuer make_prediction
        
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        
        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(float_image, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=False)
