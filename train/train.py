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

DATA_DIRECTORY = "./"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DECAY = 0.99

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate confidence score")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the testcase.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs.")
    parser.add_argument("--decay", type=int, default=DECAY,
                        help="Learning rate decay.")
    return parser.parse_args()

def main():
    args = get_arguments()
    print(args)

    path = args.data_dir
    trX = np.load(path+'train/X.npy')
    trY = np.expand_dims(np.load(path+'train/Y.npy'),1)
    vaX = np.load(path+'val/X.npy')
    vaY = np.expand_dims(np.load(path+'val/Y.npy'),1)

    tf.reset_default_graph()
    model = Decision()

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
        
        for b in trange(batchNum):
            bX, bY = next(batch)
            totalLoss += model.train(sess, bX, bY, lr)
            
        lr *= args.decay
            
        print('Epoch %d: %f' % (e+1, totalLoss/batchNum))
        if (e+1) % 10 == 0:
            print('TrainAccuracy %f' % (model.accuracy(sess, trX, trY, args.batch_size)))
        
        tmp = model.accuracy(sess, vaX, vaY, args.batch_size)
        if best > tmp:
            best = tmp
            saver.save(sess, './controlAgentModel/controlAgent.ckpt')
            save = e+1
        print('ValidAccuracy %f' % (tmp))
    saver.save(sess, './controlAgentModel/controlAgent_final.ckpt')

if __name__ == '__main__':
    main()
