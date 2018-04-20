import tensorflow as tf
import numpy as np
import random

class Decision:
    def __init__(self, feature_size=[4,8]):
        
        def lrelu(x, alpha=0.1):
            return tf.nn.relu(x)-alpha*tf.nn.relu(-x)
        
        with tf.variable_scope('controlAgent'):
            self.inputs = tf.placeholder(tf.float32, [None, feature_size[0], feature_size[1], 384])
            self.outputs = tf.placeholder(tf.float32, [None, 1])
            self.lr = tf.placeholder(tf.float32)
            self.dp = tf.placeholder(tf.float32)
            
            self.conv = tf.contrib.layers.conv2d(self.inputs, 96, [3, 3],
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        #biases_initializer=tf.constant_initializer(0.1),
                                                        activation_fn=lrelu)
            
            self.flat = tf.contrib.layers.flatten(self.conv)
            self.fc0 = tf.contrib.layers.fully_connected(self.flat, 1024,
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        biases_initializer=tf.constant_initializer(0.1),
                                                        activation_fn=lrelu)
            self.fc0 = tf.nn.dropout(self.fc0, self.dp)
            
            self.fc1 = tf.contrib.layers.fully_connected(self.fc0, 1024,
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        biases_initializer=tf.constant_initializer(0.1),
                                                        activation_fn=lrelu)
            self.fc1 = tf.nn.dropout(self.fc1, self.dp)
            
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 10,
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        biases_initializer=tf.constant_initializer(0.1),
                                                        activation_fn=lrelu)
            self.fc2 = tf.nn.dropout(self.fc2, self.dp)
            
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 1,
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        biases_initializer=tf.constant_initializer(0.1),
                                                        activation_fn=lrelu)
            
            self.loss = tf.losses.mean_squared_error(labels=self.outputs, predictions=self.fc3)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, bX, bY, lr):
        _, loss = sess.run([self.optimizer, self.loss], 
                           {self.inputs: bX, self.outputs: bY, self.lr: lr, self.dp: 0.5})
        
        return loss
    
    def pred(self, sess, bX):
        pd = sess.run(self.fc3, {self.inputs: bX, self.dp: 1.0})
        
        return pd

    def accuracy(self, sess, dX, dY, batchSize):
        batch = self.batchIterator(dX, dY, batchSize)
        batchNum = dX.shape[0]//batchSize
        acc = 0
        predy = []
        truey = []
        for b in range(batchNum):
            bX, bY = next(batch)
            pd = self.pred(sess, bX)
            predy.append(pd)
            truey.append(bY)
            acc += np.average(np.abs(pd-bY))
            
        acc /= batchNum
        
        return acc
        
    def sampleIterator(self, dX, dY):
            n = dX.shape[0]
            lst = [i for i in range(n)]
            
            while True:
                random.shuffle(lst)
                
                for i in range(n):
                    i = lst[i]
                    
                    yield dX[i], dY[i]

    def batchIterator(self, dX, dY, batchSize):
        
        sample = self.sampleIterator(dX, dY)
        
        while True:
            bX = np.zeros((batchSize, 4,8,384))
            bY = np.zeros((batchSize, 1))
            
            for i in range(batchSize):
                bX[i], bY[i] = next(sample)
            
            yield bX, bY*100