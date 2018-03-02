import numpy as np

def overlap4(index, pred, max_value, preds, preds_value, input_size=[512,1024], overlap=0):
    '''overlapping regions.
	   Args:
        index: Index of regions.
        pred: Predictions of region.
        max_value: Raw output values of region.
        preds: Predictions of whole image.
        preds_value: Raw output values of whole image.
        input_size: Image size (height, width)
        overlap: Overlapping size.
    '''
    height, width = input_size
    if index == 0:
        preds[:, :height+overlap, :width+overlap] = pred
        preds_value[:, :height+overlap, :width+overlap] = max_value

    elif index == 1:
        preds[:, :height+overlap, width+overlap:] = pred[:,:,2*overlap:]
        preds_value[:, :height+overlap, width+overlap:] = max_value[:,:,2*overlap:]

        preds[:,:height+overlap,width-overlap:width+overlap] = np.where(np.greater(max_value[:,:,:2*overlap], preds_value[:,:height+overlap,width-overlap:width+overlap]),
            pred[:,:,:2*overlap], preds[:,:height+overlap,width-overlap:width+overlap])
        preds_value[:,:height+overlap,width-overlap:width+overlap] = np.where(np.greater(max_value[:,:,:2*overlap], preds_value[:,:height+overlap,width-overlap:width+overlap]),
            max_value[:,:,:2*overlap], preds_value[:,:height+overlap,width-overlap:width+overlap])

    elif index == 2:
        preds[:, height+overlap:, :width+overlap] = pred[:,2*overlap:,:]
        preds_value[:, height+overlap:, :width+overlap] = max_value[:,2*overlap:,:]

        preds[:,height-overlap:height+overlap,:width+overlap] = np.where(np.greater(max_value[:,:2*overlap,:], preds_value[:,height-overlap:height+overlap,:width+overlap]),
            pred[:,:2*overlap,:], preds[:,height-overlap:height+overlap,:width+overlap])
        preds_value[:,height-overlap:height+overlap,:width+overlap] = np.where(np.greater(max_value[:,:2*overlap,:], preds_value[:,height-overlap:height+overlap,:width+overlap]),
            max_value[:,:2*overlap,:], preds_value[:,height-overlap:height+overlap,:width+overlap])

    elif index == 3:
        preds[:, height+overlap:, width+overlap:] = pred[:,2*overlap:,2*overlap:]
        preds_value[:, height+overlap:, width+overlap:] = max_value[:,2*overlap:,2*overlap:]

        preds[:,height-overlap:,width-overlap:width+overlap] = np.where(np.greater(max_value[:,:,:2*overlap], preds_value[:,height-overlap:,width-overlap:width+overlap]),
            pred[:,:,:2*overlap], preds[:,height-overlap:,width-overlap:width+overlap])
        preds_value[:,height-overlap:,width-overlap:width+overlap] = np.where(np.greater(max_value[:,:,:2*overlap], preds_value[:,height-overlap:,width-overlap:width+overlap]), 
            max_value[:,:,:2*overlap], preds_value[:,height-overlap:,width-overlap:width+overlap])
        preds[:,height-overlap:height+overlap,width+overlap:] = np.where(np.greater(max_value[:,:2*overlap,2*overlap:], preds_value[:,height-overlap:height+overlap,width+overlap:]),
            pred[:,:2*overlap,2*overlap:], preds[:,height-overlap:height+overlap,width+overlap:])
        #preds_value[:,height-overlap:height+overlap,width+overlap:] = np.where(np.greater(max_value[:,:2*overlap,2*overlap:], preds_value[:,height-overlap:height+overlap,width+overlap:]), 
        #   max_value[:,:2*overlap,2*overlap:], preds_value[:,height-overlap:height+overlap,width+overlap:])

def overlap2(index, pred, max_value, preds, preds_value, input_size=[1024,1024], overlap=0):
    height, width = input_size
    if index == 0:
        preds[:, :, :width+overlap] = pred
        preds_value[:, :, :width+overlap] = max_value

    elif index == 1:
        preds[:, :, width+overlap:] = pred[:,:,2*overlap:]
        preds_value[:, :, width+overlap:] = max_value[:,:,2*overlap:]

        preds[:,:,width-overlap:width+overlap] = np.where(np.greater(max_value[:,:,:2*overlap], preds_value[:,:,width-overlap:width+overlap]),
            pred[:,:,:2*overlap], preds[:,:,width-overlap:width+overlap])
        preds_value[:,:,width-overlap:width+overlap] = np.where(np.greater(max_value[:,:,:2*overlap], preds_value[:,:,width-overlap:width+overlap]),
            max_value[:,:,:2*overlap], preds_value[:,:,width-overlap:width+overlap])

