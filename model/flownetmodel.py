from tools.flow_utils import LeakyReLU, pad, antipad
import math
import tensorflow as tf
slim = tf.contrib.slim

class FlowNets(object):
	def __init__(self, images1, images2, is_training=False):
		with tf.variable_scope('FlowNets'):
			images= tf.concat([images1, images2], 3)
			with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
								# Only backprop this network if trainable
								trainable=is_training,
								# He (aka MSRA) weight initialization
								weights_initializer=slim.variance_scaling_initializer(),
								#biases_initializer=tf.zeros_initializer(dtype=tf.float16),
								activation_fn=LeakyReLU,
								# We will do our own padding to match the original Caffe code
								padding='VALID'):

				weights_regularizer = slim.l2_regularizer(0.0004)
				with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
					with slim.arg_scope([slim.conv2d], stride=2):
						conv1 = slim.conv2d(pad(images, 3), 24, 7, scope='conv1')
						conv2 = slim.conv2d(pad(conv1, 2), 48, 5, scope='conv2')
						conv3 = slim.conv2d(pad(conv2, 2), 96, 5, scope='conv3')

					conv3_1 = slim.conv2d(pad(conv3), 96, 3, scope='conv3_1')
					with slim.arg_scope([slim.conv2d], num_outputs=192, kernel_size=3):
						conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
						conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
						conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
						conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
					conv6 = slim.conv2d(pad(conv5_1), 384, 3, stride=2, scope='conv6')
					conv6_1 = slim.conv2d(pad(conv6), 384, 3, scope='conv6_1')

					""" START: Refinement Network """
					with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
						predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,
													scope='predict_flow6',
													activation_fn=None)
						deconv5 = antipad(slim.conv2d_transpose(conv6_1, 192, 4,
																stride=2,
																scope='deconv5')
																,evenh = conv5_1.shape.as_list()[1]%2==0
																,evenw = conv5_1.shape.as_list()[2]%2==0)
						upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,
																		stride=2,
																		scope='upsample_flow6to5',
																		activation_fn=None)
																		,evenh = conv5_1.shape.as_list()[1]%2==0
																		,evenw = conv5_1.shape.as_list()[2]%2==0)
						concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

						predict_flow5 = slim.conv2d(pad(concat5), 2, 3,
													scope='predict_flow5',
													activation_fn=None)
						deconv4 = antipad(slim.conv2d_transpose(concat5, 96, 4,
																stride=2,
																scope='deconv4')
																,evenh = conv4_1.shape.as_list()[1]%2==0
																,evenw = conv4_1.shape.as_list()[2]%2==0)
						upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,
																		stride=2,
																		scope='upsample_flow5to4',
																		activation_fn=None)
																		,evenh = conv4_1.shape.as_list()[1]%2==0
																		,evenw = conv4_1.shape.as_list()[2]%2==0)
						concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

						predict_flow4 = slim.conv2d(pad(concat4), 2, 3,
													scope='predict_flow4',
													activation_fn=None)
						deconv3 = antipad(slim.conv2d_transpose(concat4, 48, 4,
																stride=2,
																scope='deconv3')
																,evenh = conv3_1.shape.as_list()[1]%2==0
																,evenw = conv3_1.shape.as_list()[2]%2==0)
						upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
																		stride=2,
																		scope='upsample_flow4to3',
																		activation_fn=None)
																		,evenh = conv3_1.shape.as_list()[1]%2==0
																		,evenw = conv3_1.shape.as_list()[2]%2==0)
						concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

						predict_flow3 = slim.conv2d(pad(concat3), 2, 3,
													scope='predict_flow3',
													activation_fn=None)
						deconv2 = antipad(slim.conv2d_transpose(concat3, 24, 4,
																stride=2,
																scope='deconv2')
																,evenh = conv2.shape.as_list()[1]%2==0
																,evenw = conv2.shape.as_list()[2]%2==0)
						upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
																		stride=2,
																		scope='upsample_flow3to2',
																		activation_fn=None)
																		,evenh = conv2.shape.as_list()[1]%2==0
																		,evenw = conv2.shape.as_list()[2]%2==0)
						concat2 = tf.concat([conv2, deconv2, upsample_flow3to2], axis=3)

						predict_flow2 = slim.conv2d(pad(concat2), 2, 3,
													scope='predict_flow2',
													activation_fn=None)
						""" END: Refinement Network """
						self.scale = slim.conv2d(pad(concat2), 19, 3,
													weights_initializer=tf.constant_initializer(0.0),
													biases_initializer=tf.constant_initializer(1.0),
													scope='predict_scale',
													activation_fn=None)

				self.flow = predict_flow2 * 5.0
				self.feature = conv6_1
				# TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
	def inference(self):
		return {
			'feature': self.feature,
			'scale': self.scale,
			'flow': self.flow,
		}
