import os
import random
import tensorflow as tf
import numpy as np

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    for line in f:
        try:
            image = line[:-1].split('\n')[0]
        except ValueError: # Adhoc for test.
            image = line.strip("\n")

        image = data_dir+image
        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image1)

        images.append(image)
    f.close()
    return images


def read_images_from_disk(input_queue, input_size, overlap, img_mean=IMG_MEAN):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
        filename_tensor: A scalar string tensor.
    Returns:
        Three tensors: the decoded images and flos.
    """
    height = input_size[0]//2
    height_overlap = height+overlap
    width = input_size[1]//2
    width_overlap = width+overlap

    image_file = tf.read_file(input_queue[0])
    image = tf.image.decode_image(image_file)

    image = tf.cast(image,tf.float32)

    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
    image_bgr = tf.concat(axis=2, values=[img_b, img_g, img_r])
    image_bgr.set_shape((None, None, 3))
    image_bgr = tf.expand_dims(tf.image.resize_images(image_bgr, input_size), 0)

    images = tf.concat([image_bgr[:, :height+overlap, :width+overlap, :],
                    image_bgr[:, :height+overlap, width-overlap:, :],
                    image_bgr[:, height-overlap:, :width+overlap, :],
                    image_bgr[:, height-overlap:, width-overlap:, :]],0)

    # Preprocess.
    image_s = images-img_mean
    image_f = tf.image.resize_images(images/255.0, [(height_overlap)//2, (width_overlap)//2])

    return image_s, image_f

def _generate_image_and_label_batch(image_s, image_f, batch_size):
    """Construct a queued batch of images and labels.

    Args:
        image_s, image_f: 3-D Tensor of input image of type.float32.
        batch_size: Number of images per batch.

    Returns:
        bimages: Images. 4D tensor of [batch_size, height, width, 3] size.
        bflo: Flos. 4D tensor of [batch_size, height, width, 2] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    bimage_s, bimage_f = tf.train.batch(
            [image_s, image_f],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=1)

    return bimage_s, bimage_f

def inputs(data_dir, data_list, batch_size, input_size, overlap, img_mean=IMG_MEAN):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        data_dir: Path to the FlowNet data directory.
        batch_size: Number of images per batch.

    Returns:
        image1, image2: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    """
    image_list = read_labeled_image_list(data_dir, data_list)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([images], shuffle=False)
    image_s, image_f = read_images_from_disk(input_queue, input_size, overlap, img_mean)
    
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image_s, image_f, batch_size)
