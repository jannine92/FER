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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import cv2
from matplotlib import pyplot as plt

import tensorflow as tf
import fer2013



FLAGS = tf.app.flags.FLAGS

local_directory = os.path.dirname(os.path.abspath(__file__))+ '/fer2013' + '/'


tf.app.flags.DEFINE_string('eval_dir', (local_directory+'eval'),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'make_prediction',
                           """Either 'test' or 'train_eval' or 'make_prediction'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', (local_directory+'train'), #actually: 'train'
                           """Directory where to read model checkpoints.""")


# own image taken by webcam (see webcam_prediction.py)
input_image_csv = 'Images/image1.csv'
input_image_png = 'Images/image1.png'

emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5:'Surprise', 6: 'Neutral'}




def predict():
    """Eval FER2013 for a number of steps."""
    with tf.Graph().as_default():


        images, _ = fer2013.inputs(eval_data=FLAGS.eval_data, input_file=input_image_csv)
        keep_prob = 1
        
        
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = fer2013.inference(images, keep_prob, -1) # batch size -1: accepts dynamic batch sizes

        
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            fer2013.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        
        # ---------------------- Version 1 with top k predictions --------------------
        _, top_k_pred = tf.nn.top_k(logits, k=3)
        
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("checkpoint file found")
            else:
                print('No checkpoint file found')
                return
                
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            top_indices = sess.run([top_k_pred])
            print ("Predicted as TOP 3: ", top_indices[0][0], " for your input image.")
            print('Predicted as 1.: ', top_indices[0][0][0], ' -> ', emotion_dict[top_indices[0][0][0]])
            print('Predicted as 2.: ', top_indices[0][0][1], ' -> ', emotion_dict[top_indices[0][0][1]])
            print('Predicted as 3.: ', top_indices[0][0][2], ' -> ', emotion_dict[top_indices[0][0][2]])
            
            coord.request_stop()  
            coord.join(threads)
        
        # -------------------- end Version 1 -----------------------
        
        # Version 2 with argmax
        # Restore the moving average version of the learned variables for eval.

        """
        prediction = make_prediction(saver=saver, logits=logits)
        print('Predicted emotion: ', prediction[0], ' -> ', emotion_dict[prediction[0]])
        """
        
        
        # ------------------ end Version 2 --------------------
        
        img = cv2.imread(local_directory + input_image_png, 0)
        cv2.imshow("Emotion Image", img)
        print("--------------- Press ENTER to return to Webcam ---------------")

        key = cv2.waitKey(0)
        if key == 13:
            cv2.destroyAllWindows()
        

def make_prediction(saver, logits):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("checkpoint file found")
        else:
            print('No checkpoint file found')
            return
        
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        highest_prob = tf.argmax(logits, 1)
        prediction = highest_prob.eval()
        
        coord.request_stop()  
        coord.join(threads)
        
        return prediction


def main(input_image=None, webcam=False): 
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    if webcam:
        global input_image_csv
        global input_image_png
        input_image_csv = input_image + '.csv'
        input_image_png = input_image + '.png'
    predict()


if __name__ == '__main__':
    tf.app.run()
