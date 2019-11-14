#!/usr/bin/env python

import numpy as np
import cv2
import sys

img = cv2.imread(sys.argv[1])
img2 = np.stack([img[:,:,0], img[:,:,1], img[:,:,2], 0*img[:,:,0]+255], axis=2)

def get_block(i, orig_image=True):
    """Get a 8x8 block of the image, counting horizontally
    The image has 640/8=80 block horizontally and 480/8=60 blocks vertically

    parms:
        i (int): block number, should be between 0 and 80*60=4800
    """
    if i<0 or i>=4800:
        raise ValueError("Invalid block number: " + str(i))
    
    rownum = i // 80
    colnum = i % 80
    if orig_image:
        return img[rownum*8:(rownum+1)*8, colnum*8:(colnum+1)*8]
    else:
        return img2[rownum*8:(rownum+1)*8, colnum*8:(colnum+1)*8]

def is_constant(block):
    return np.all(block == block[0,0])

def is_flagged(i):
    """
    Determine whether an 8x8 block should be considered as a missing block.
    The following heuristic is used: the 8x8 block should have a constant value,
    and should be totally equal to MIN_CHUNK_LENGTH of the surrounding 8x8 blocks
    """
    MIN_CHUNK_LENGTH = 12
    block = get_block(i)
    if is_constant(block):
        num_neighborflags = 0
        for neighbor_num in range(max(i-MIN_CHUNK_LENGTH,0), min(i+MIN_CHUNK_LENGTH+1,4799)):
            if neighbor_num == i:
                continue
            neighbor_block = get_block(neighbor_num)
            if is_constant(neighbor_block) and np.all(neighbor_block[0,0] == block[0,0]):
                num_neighborflags += 1
        if num_neighborflags >= MIN_CHUNK_LENGTH:
            return True
    return False

for i in range(80*60):
    if is_flagged(i):
        block2 = get_block(i, orig_image = False)
        block2[:,:,3] = 0

cv2.imwrite(sys.argv[1].rstrip(".jpeg") + ".png", img2)
