from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import cv2
from tqdm import trange

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from model import DeepLab_Fast
from tools.img_utils import preprocess
from tools.flow_utils import warp

DATA_DIRECTORY = '/data/cityscapes_dataset/cityscape/'
DATA_LIST_PATH = '../list/train_list.txt'
RESTORE_FROM = '../checkpoint/'
SAVE_DIR = '/data/cityscapes_dataset/cityscape/decision/real/train/'
NUM_CLASSES = 19
NUM_STEPS = 2975 # Number of images in the dataset.
input_size = [1024, 2048]
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate real testcases")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the video.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    print(args)

    tf.reset_default_graph()

    # Set placeholder 
    image_filename = tf.placeholder(dtype=tf.string)

    # Read & Decode image
    image = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    image.set_shape([None, None, 3])
    image = tf.expand_dims(preprocess(image), dim=0)
    image_batch = tf.image.resize_bilinear(image-IMG_MEAN, input_size)
    output_size = [512, 1024]
    image_batch = tf.concat([image_batch[:,:512,:1024,:],image_batch[:,:512,1024:,:],image_batch[:,512:,:1024,:],image_batch[:,512:,1024:,:]],0)

    # Create network.
    net = DeepLab_Fast({'data': image_batch}, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_pred = net.layers['fc_out']
    raw_output = tf.image.resize_bilinear(raw_pred, output_size)
    output = tf.cast(tf.argmax(raw_output, axis=3), tf.uint8)
        
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
        
    # Load weights.
    ckpt = tf.train.get_checkpoint_state(args.restore_from)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    list_file = open(args.data_list, 'r')

    for step in trange(args.num_steps):
        f1, f2 = list_file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(args.data_dir, f1)

        pred = sess.run(output, feed_dict={image_filename:f1})
        
        filename = f1.split('/')[7].replace("leftImg8bit.png","")
        for i in xrange(4):
            cv2.imwrite(args.save_dir + filename + 'pred_' + str(i) + '.png', pred[i])

    print("Generate real testcases finish!")
if __name__ == '__main__':
    main()
