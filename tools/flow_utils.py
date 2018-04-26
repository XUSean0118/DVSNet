import tensorflow as tf
import numpy as np
import time

# Thanks, https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, leak=0.1, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1.0 + leak)
		f2 = 0.5 * (1.0 - leak)
		return f1 * x + f2 * abs(x)


def average_endpoint_error(labels, predictions):
	"""
	Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
		sqrt[sum_across_channels{(X - Y)^2}]
	"""
	num_samples = predictions.shape.as_list()[0]
	with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
		predictions = tf.to_float(predictions)
		labels = tf.to_float(labels)
		predictions.get_shape().assert_is_compatible_with(labels.get_shape())

		squared_difference = tf.square(tf.subtract(predictions, labels))
		# sum across channels: sum[(X - Y)^2] -> N, H, W, 1
		loss = tf.reduce_sum(squared_difference, 3, keep_dims=True)
		loss = tf.sqrt(loss)
		return tf.reduce_sum(loss) / num_samples


def pad(tensor, num=1):
	"""
	Pads the given tensor along the height and width dimensions with `num` 0s on each side
	"""
	return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, evenh=True, evenw=True, num=1):
	"""
	Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
	padding from the output rather than adding it to the input.
	"""
	batch, h, w, c = tensor.shape.as_list()
	if evenh and evenw:
		return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])
	elif evenh and (not evenw):
		return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num , w - 2 * num- 1, c])
	elif (not evenh) and evenw:
		return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num - 1, w - 2 * num, c])
	else:
		return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num - 1, w - 2 * num - 1, c])

def get_xyindex(h,w):
	index_list = []
	for i in xrange(h):
		for j in xrange(w):
			index_list.append([j,i])
	return np.array(index_list)

def get_batchindex(b,h,w):
	index_list = []
	for k in xrange(b):
		for i in xrange(h):
			for j in xrange(w):
				index_list.append([k])
	return np.array(index_list)

def warp(key_feature, flow):
	shape = flow.get_shape().as_list()
	key_feature = tf.image.resize_bilinear(key_feature, shape[1:3])
	batch_size = shape[0]
	height = shape[1]
	width = shape[2]
	with tf.name_scope('warp') as scope:
		flow_index = flow + tf.constant(get_xyindex(height, width),shape=[height, width, 2],dtype=tf.float32)
		flow_index = tf.minimum(flow_index, [width-1,height-1])
		flow_index = tf.maximum(flow_index, [0,0])
		batch_index = tf.constant(get_batchindex(batch_size, height, width),shape=[batch_size, height, width, 1],dtype=tf.float32)
		x_index = tf.reshape(flow_index[:,:,:,0], [batch_size, height, width, 1])
		y_index = tf.reshape(flow_index[:,:,:,1], [batch_size, height, width, 1])
		x_floor = tf.floor(x_index)
		x_ceil = tf.ceil(x_index)
		y_floor = tf.floor(y_index)
		y_ceil = tf.ceil(y_index)
		flow_index_ff = tf.cast(tf.concat([batch_index,y_floor,x_floor], 3), tf.int32)
		flow_index_cf = tf.cast(tf.concat([batch_index,y_ceil,x_floor], 3), tf.int32)
		flow_index_fc = tf.cast(tf.concat([batch_index,y_floor,x_ceil], 3), tf.int32)
		flow_index_cc = tf.cast(tf.concat([batch_index,y_ceil,x_ceil], 3), tf.int32)
		thetax = x_index - x_floor
		_thetax = 1.0 - thetax
		thetay =  y_index - y_floor
		_thetay = 1.0 - thetay
		coeff_ff = _thetax * _thetay
		coeff_cf = _thetax * thetay
		coeff_fc = thetax * _thetay
		coeff_cc = thetax * thetay
		ff = tf.gather_nd(key_feature, flow_index_ff) * coeff_ff
		cf = tf.gather_nd(key_feature, flow_index_cf) * coeff_cf
		fc = tf.gather_nd(key_feature, flow_index_fc) * coeff_fc
		cc = tf.gather_nd(key_feature, flow_index_cc) * coeff_cc
		warp_image = tf.add_n([ff,cf,fc,cc])
	return warp_image

