"""Class prediction and error visualization helpers."""
import numpy as np

CLASS_NAMES=['Avicennia','Rhizophora','Mixed Mangrove','Aquaculture','Buildings','Water']
CLASS_COLORS=np.array([[45,160,75],[0,100,0],[166,217,106],
                       [255,165,0],[220,20,60],[30,144,255]],dtype=np.uint8)


def colorize(arr):
    rgb=np.full((*arr.shape,3),255,dtype=np.uint8)
    valid=(arr>=0)&(arr<6)
    rgb[valid]=CLASS_COLORS[arr[valid]]
    return rgb


def error_map(pred, gt):
    """Gray correct, predicted-class color for wrong pixels, white ignored."""
    rgb=np.full((*gt.shape,3),255,dtype=np.uint8)
    valid=gt!=255
    rgb[valid & (pred==gt)]=[190,190,190]
    wrong=valid & (pred!=gt)
    rgb[wrong]=CLASS_COLORS[pred[wrong]]
    return rgb
