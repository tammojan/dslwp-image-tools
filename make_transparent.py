#!/usr/bin/env python

import numpy as np
import cv2
import sys
from argparse import ArgumentParser


def add_alpha_channel(img):
    if img.shape[-1] != 3:
        raise(ValueError("Expecting image with 3 layers (r,g,b), got " + str(img.shape[1])))
    return np.stack([img[:,:,0], img[:,:,1], img[:,:,2], 0*img[:,:,0]+255], axis=2)

def get_block(img, i):
    """Get a 8x8 block of the image, counting horizontally
    The image has 640/8=80 block horizontally and 480/8=60 blocks vertically

    parms:
        i (int): block number, should be between 0 and 80*60=4800
    """
    if i<0 or i>=4800:
        raise ValueError("Invalid block number: " + str(i))
    
    rownum = i // 80
    colnum = i % 80
    return img[rownum*8:(rownum+1)*8, colnum*8:(colnum+1)*8]

def is_constant(blocknum, block, ignore_boundary=False):
    if ignore_boundary and blocknum%80 == 79:
        # ignore last column, tolerance for jpeg artefacts
        return np.all(np.abs(block[:,:7,:3] - block[1,1,:3]) < 4)
    return np.all(block[:,:,:3] == block[1,1,:3])

def is_missingchunk(img, i):
    """
    Determine whether an 8x8 block should be considered as a missing block.
    The following heuristic is used: the 8x8 block should have a constant value,
    and should be totally equal to MIN_CHUNK_LENGTH of the surrounding 8x8 blocks
    """
    MIN_CHUNK_LENGTH = 12
    block = get_block(img, i)
    if is_constant(i, block) and np.all(block[1,1,:3]<250):
        # Heuristic: fully overexposed blocks are not missing chunks
        num_neighborflags = 0
        for neighbor_num in range(max(i-MIN_CHUNK_LENGTH,0), min(i+MIN_CHUNK_LENGTH+1,4799)):
            if neighbor_num == i:
                continue
            neighbor_block = get_block(img, neighbor_num)
            if is_constant(neighbor_num, neighbor_block) and \
                   np.all(neighbor_block[1,1,:3] == block[1,1,:3]):
                num_neighborflags += 1
        if num_neighborflags >= MIN_CHUNK_LENGTH:
            return True
    return False

def make_missing_chunks_transparent(img):
    if img.shape[-1] != 4:
        img = add_alpha_channel(img)

    for i in range(80*60):
        if is_missingchunk(img, i):
            block = get_block(img, i)
            block[:,:,3] = 0

    return img

def make_black_blocks_transparent(img):
    """Makes constant black blocks transparent"""
    if img.shape[-1] != 4:
        img = add_alpha_channel(img)

    for i in range(80*60):
        block = get_block(img, i)
        if is_constant(i, block, ignore_boundary=True):
            if np.all(block[:,:7,:3] < 5):
                block[:,:,3] = 0

    return img

if __name__ == "__main__":
    parser = ArgumentParser(description="Add transparency to DSLWP images")
    parser.add_argument("-b", "--remove_black", action="store_true", help="Remove black blocks")
    parser.add_argument("image", help="Image to clean up")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    img = make_missing_chunks_transparent(img)
    if args.remove_black:
        img = make_black_blocks_transparent(img)
    cv2.imwrite(args.image.rstrip(".jpeg") + ".png", img)
