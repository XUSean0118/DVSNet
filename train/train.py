from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import trange

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from model import Decision

TRAIN_DATA_DIRECTORY = "./train/"
VAL_DATA_DIRECTORY = "./val/"
SAVE_DIR = './decisionModel/'
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.002
DECAY = 0.99

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train Decision Network")
    parser.add_argument("--train_data_dir", type=str, default=TRAIN_DATA_DIRECTORY,
                        help="Path to the directory containing the train testcase.")
    parser.add_argument("--val_data_dir", type=str, default=VAL_DATA_DIRECTORY,
                        help="Path to the directory containing the validation testcase.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save decision model.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs.")
    parser.add_argument("--decay", type=int, default=DECAY,
                        help="Learning rate decay.")
    return parser.parse_args()

def main():
    args = get_arguments()
    print(args)

    trpath = args.train_data_dir
    vapath = args.val_data_dir
    trX = np.load(trpath+'X.npy')
    trY = np.expand_dims(np.load(trpath+'Y.npy'),1)
    vaX = np.load(vapath+'X.npy')
    vaY = np.expand_dims(np.load(vapath+'Y.npy'),1)

    tf.reset_default_graph()
    model = Decision(is_training = True)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    # Set up input
    batch = model.batchIterator(trX, trY, args.batch_size)
    batchNum = trX.shape[0]//args.batch_size
    lr = args.learning_rate

    best = 100
    save = 0
    for e in range(args.epochs):
        totalLoss = 0
        
        for b in range(batchNum):
            bX, bY = next(batch)
            totalLoss += model.train(sess, bX, bY, lr)
            
        lr *= args.decay
            
        print('Epoch %d: %f' % (e+1, totalLoss/batchNum))
        if (e+1) % 10 == 0:
            print('TrainAccuracy %f' % (model.accuracy(sess, trX, trY, args.batch_size)))
        
        tmp = model.accuracy(sess, vaX, vaY, args.batch_size)
        if best > tmp:
            best = tmp
            saver.save(sess,SAVE_DIR+'model.ckpt')
            save = e+1
        print('ValidAccuracy %f' % (tmp))
    saver.save(sess, SAVE_DIR+'modelFinall.ckpt')

if __name__ == '__main__':
    main()
