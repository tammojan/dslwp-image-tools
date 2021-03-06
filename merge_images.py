#!/usr/bin/env python

import numpy as np
import cv2
from make_transparent import make_missing_chunks_transparent, get_block
from argparse import ArgumentParser

def merge_images(images):
    """
    Merges images that fix missing chunks

    Args:
        images (List[np.array]): list of input images with alpha channel

    Returns:
        np.array: merged image
    """
    for image in images:
        if image.shape[-1] != 4:
            raise ValueError("Image should have alpha channel")

    output_image = images[0].copy()

    for image in images[1:]:
        for blocknum in range(80*60):
            output_block = get_block(output_image, blocknum)
            if output_block[0,0,3] == 0:
                new_block = get_block(image, blocknum)
                output_block[:,:,:] = new_block

    return output_image

def add_black_background(image):
    """Add a black background for transparent blocks"""

    image = image.copy()
    for blocknum in range(80*60):
        block = get_block(image, blocknum)
        if block[0,0,3] == 0:
            block[:,:,:3] = 0
            block[:,:,3] = 255

    return image


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge two DSLWP images")
    parser.add_argument("input", help="Input images", nargs="+")
    parser.add_argument("-o", "--output", help="Output image name (default: out.png)", type=str, default="out.png")
    parser.add_argument("-b", "--black-background", help="Add black background", action="store_true")

    args = parser.parse_args()

    input_images = []
    for inputname in args.input:
        input_image = cv2.imread(inputname)
        input_image = make_missing_chunks_transparent(input_image)
        input_images += [input_image]

    output_image = merge_images(input_images)

    if args.black_background:
        output_image = add_black_background(output_image)

    cv2.imwrite(args.output, output_image)
