from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

REAL_DATA_DIRECTORY = "/data/cityscapes_dataset/cityscape/decision/real/"
PRED_DATA_DIRECTORY = "/data/cityscapes_dataset/cityscape/decision/pred/"
TRAIN_LIST_PATH = "../list/imagestrain_13_list.txt"
VAL_LIST_PATH = "../list/imagesval_10_list.txt"
SAVE_DIR = "./"

class confidence:
    def __init__(self):
        self.INVALID = 255

        tf.reset_default_graph()
        self.pdIn = tf.placeholder(dtype=tf.float32, shape=[512*1024]) 
        self.gtIn = tf.placeholder(dtype=tf.float32, shape=[512*1024]) 
        self.wight = tf.where(tf.equal(self.gtIn, self.INVALID),tf.zeros_like(self.gtIn),tf.ones_like(self.gtIn))

        self.accuracy = tf.where(tf.equal(self.pdIn, self.gtIn),self.wight,tf.zeros_like(self.gtIn))

        self.average = tf.reduce_sum(self.accuracy)/tf.reduce_sum(self.wight)

        self.sess = tf.Session()
    
    def cal(self, pd, gt):
        pd = pd.flatten()
        gt = gt.flatten()
        accuracy = self.sess.run(self.average, {self.pdIn: pd, self.gtIn: gt})
        return accuracy

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate confidence score")
    parser.add_argument("--real-data-dir", type=str, default=REAL_DATA_DIRECTORY,
                        help="Path to the directory containing the real testcase.")
    parser.add_argument("--pred-data-dir", type=str, default=PRED_DATA_DIRECTORY,
                        help="Path to the directory containing the pred testcase.")
    parser.add_argument("--train-list", type=str, default=TRAIN_LIST_PATH,
                        help="Path to the file listing the data in the train testcase.")
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the data in the val testcase.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save testcase .npy.")
    parser.add_argument("--clip", type=float, default=0.0,
                        help="trim extreme confidence score")
    return parser.parse_args()

def main():
    args = get_arguments()
    print(args)

    model = confidence()

    for dataset in ["train", "val"]:
        filenames = []
        if dataset == "train":
            list_file = open(args.train_list, 'r') 
        else:
            list_file = open(args.val_list, 'r') 
        for line in list_file:
            line = line[:-1]
            filenames.append(line.split(' '))
        list_file.close()

        # load pred and real testcase and calculate confidence score
        # store confidence score and feature
        score_list = []
        ft_list = []
        real_path = os.path.join(args.real_data_dir,dataset)
        pred_path = os.path.join(args.pred_data_dir,dataset)
        for file in tqdm(filenames):
            f0 = file[0][file[0].rfind('/')+1:].replace('leftImg8bit', '{ID}')
            f1 = file[2][file[2].rfind('/')+1:].replace('gtFine_labelTrainIds', '{ID}')
        
            for i in range(4):
                f = f0.replace('{ID}', 'pred_%d'%(i))
                img0 = cv2.imread(os.path.join(pred_path,f), 0)
                f = f1.replace('{ID}', 'pred_%d'%(i))
                img1 = cv2.imread(os.path.join(real_path,f), 0)
                f = f0.replace('{ID}', 'flowfeature_%d'%(i)).replace('png', 'npy')
                ft = np.load(os.path.join(pred_path,f))
                
                score = model.cal(img0, img1)
                if score > args.clip:
                    ft_list.append(ft)
                    score_list.append(score)
                    
        # save confidence score and feature
        np.save(args.save_dir+dataset+"/X", ft_list) 
        np.save(args.save_dir+dataset+"/Y", score_list)

if __name__ == '__main__':
    main()
