from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from model import DeepLab_Fast, FlowNets
from tools.img_utils import preprocess
from tools.flow_utils import warp

DATA_DIRECTORY = '/data/cityscapes_dataset/cityscape/'
DATA_LIST_PATH = '../list/imagestrain_13_list.txt'
RESTORE_FROM = '../checkpoint/'
SAVE_DIR = './train/'
NUM_CLASSES = 19
NUM_STEPS = 38675 # Number of images in the dataset.
input_size = [1024,2048]
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate predict testcases")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the video.")
    parser.add_argument("--clip", type=float, default=80.0,
                        help="trim extreme confidence score")
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
    image1_filename = tf.placeholder(dtype=tf.string)
    image2_filename = tf.placeholder(dtype=tf.string)
    
    # Read & Decode image
    image1 = tf.image.decode_image(tf.read_file(image1_filename), channels=3)
    image2 = tf.image.decode_image(tf.read_file(image2_filename), channels=3)
    image1.set_shape([None, None, 3])
    image2.set_shape([None, None, 3])
    image1 = tf.expand_dims(preprocess(image1), dim=0)
    image2 = tf.expand_dims(preprocess(image2), dim=0)
    image_batch = tf.image.resize_bilinear(image1-IMG_MEAN, input_size)

    current_frame = tf.image.resize_bilinear((image2)/255.0, (input_size[0]//2, input_size[1]//2))
    key_frame = tf.image.resize_bilinear((image1)/255.0, (input_size[0]//2, input_size[1]//2))

    output_size = [512,1024]
    image_batch = tf.concat([image_batch[:,:512,:1024,:],image_batch[:,:512,1024:,:],image_batch[:,512:,:1024,:],image_batch[:,512:,1024:,:]],0)
    key_frame = tf.concat([key_frame[:,:256,:512,:],key_frame[:,:256,512:,:],key_frame[:,256:,:512,:],key_frame[:,256:,512:,:]],0)
    current_frame = tf.concat([current_frame[:,:256,:512,:],current_frame[:,:256,512:,:],current_frame[:,256:,:512,:],current_frame[:,256:,512:,:]],0)

    # Create network.
    net = DeepLab_Fast({'data': image_batch}, num_classes=NUM_CLASSES)
    flowNet = FlowNets(current_frame, key_frame)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_pred = net.layers['fc_out']
    raw_output = tf.image.resize_bilinear(raw_pred, output_size)
    raw_output = tf.argmax(raw_output, axis=3)

    flows = flowNet.inference()
    warp_pred = warp(tf.image.resize_bilinear(raw_pred, flows['flow'].get_shape()[1:3]), flows['flow'])
    scale_pred = tf.multiply(warp_pred, flows['scale'])
    wrap_output = tf.image.resize_bilinear(scale_pred, output_size)
    output = tf.argmax(wrap_output, axis=3)

    # Calculate confidence score.
    wight = tf.where(tf.equal(raw_output, 255),tf.zeros_like(raw_output),tf.ones_like(raw_output))
    accuracy = tf.where(tf.equal(output, raw_output),wight,tf.zeros_like(raw_output))
    average = tf.divide(tf.reduce_sum(tf.contrib.layers.flatten(accuracy), 1),tf.reduce_sum(tf.contrib.layers.flatten(wight), 1))

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

    score_list = []
    ft_list = []
    for step in range(args.num_steps):
        f1, f2, f3 = list_file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(args.data_dir, f1)
        f2 = os.path.join(args.data_dir, f2)

        flow_feature, seg_feature, score = sess.run([flows['feature'], net.layers['fc1_voc12'], average],
                        feed_dict={image1_filename:f1, image2_filename:f2})

        for i in range(4):
            if score[i] > args.clip:
                ft_list.append(flow_feature[i])
                score_list.append(score[i])

        if step % 100 == 0:
            print(step)
    # save confidence score and feature
    np.save(args.save_dir+"X", ft_list) 
    np.save(args.save_dir+"Y", score_list)

    print("Generate finish!")
if __name__ == '__main__':
    main()
