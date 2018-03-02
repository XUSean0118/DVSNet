import numpy as np
import tensorflow as tf
import os

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69)
                # 0 = road, 1 = sidewalk, 2 = building
                ,(102, 102, 156), (190, 153, 153), (153, 153, 153)
                # 3 = wall, 4 = fence, 5 = pole
                ,(250, 170, 29), (219, 219, 0), (106, 142, 35)
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,(152, 250, 152), (69, 129, 180), (219, 19, 60)
                # 9 = terrain, 10 = sky, 11 = person
                ,(255, 0, 0), (0, 0, 142), (0, 0, 69)
                # 12 = rider, 13 = car, 14 = truck
                ,(0, 60, 100), (0, 79, 100), (0, 0, 230)
                # 15 = bus, 16 = train, 17 = motocycle
                ,(119, 10, 32)]
                # 18 = bicycle

def decode_labels(mask, num_classes):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    color_table = label_colours

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    outputs = tf.matmul(onehot_output, color_mat)
    outputs = tf.cast(tf.reshape(outputs, (n, h, w, 3)), dtype=tf.uint8)
    
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=False):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.
    Returns:
      Outputs a tensor of shape [batch_size h w num_classes]
      with last dimension comprised of 0's and 1's only if one_hot=True.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch

def preprocess(img):
    """ Convert RGB to BGR
    
    Args:
      img: input RGB images.

    Returns:
      BGR input images 3-D.
    """
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)

    return img

def inv_preprocess(imgs):
    """Inverse preprocessing of the batch of images.
       convert from BGR to RGB.
       
    Args:
      imgs: batch of BGR input images.
  
    Returns:
      The batch of RGB input images 4-D.
    """
    img_b, img_g, img_r = tf.split(axis=3, num_or_size_splits=3, value=imgs)
    imgs = tf.cast(tf.concat(axis=3, values=[img_r, img_g, img_b]), dtype=tf.uint8)

    return imgs
