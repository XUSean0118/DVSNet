import tensorflow as tf
import numpy as np

class Decision:
	def __init__(self, feature_size=[4,8]):
		
		def lrelu(x, alpha=0.1):
			return tf.nn.relu(x)-alpha*tf.nn.relu(-x)
		
		with tf.variable_scope('controlAgent'):
			self.inputs = tf.placeholder(tf.float32, [None, feature_size[0], feature_size[1], 384])
			self.outputs = tf.placeholder(tf.float32, [None, 1])
			self.dp = tf.placeholder(tf.float32)

			self.conv = tf.contrib.layers.conv2d(self.inputs, 96, [3, 3], activation_fn=lrelu)
			self.conv = tf.nn.dropout(self.conv, self.dp)
			self.flat = tf.contrib.layers.flatten(self.conv)
			self.fc0 = tf.contrib.layers.fully_connected(self.flat, 1024, activation_fn=lrelu)
			self.fc0 = tf.nn.dropout(self.fc0, self.dp)
			self.fc1 = tf.contrib.layers.fully_connected(self.fc0, 1024, activation_fn=lrelu)
			self.fc1 = tf.nn.dropout(self.fc1, self.dp)
			self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 10, activation_fn=lrelu)
			self.fc2 = tf.nn.dropout(self.fc2, self.dp)
			self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 1, activation_fn=lrelu)
			
	def pred(self, sess, bX):
		pd = sess.run(self.fc3, {self.inputs: bX, self.dp: 1.0})
		
		return pd
