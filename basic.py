# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
import numpy
import sys
import math
import time
from PIL import Image


#################
# Image helpers #
#################

def save_image(arr, filename, color="RGB"):
    """Takes a numpy array, in proper shape, and writes is an image
    Args:
        arr     : numpy array in shape (x, y, rgb)
        filename: name of the image to write
    """

    img = Image.fromarray(arr.astype(numpy.uint8), color)
    img.save(filename)


def image_to_array(file):
    """Read an image into a numpy 3d array
    Args:
        file: filepath to image
    Returns:
        A 3d numpy array of type uint8.
    """
    img = Image.open(file)
    img_arr =  numpy.asarray(img).astype(numpy.int32)
    return img_arr


# Turn the image into a grayscale image (i.e., 1 byte per pixel).
def convert_to_grayscale(arr):
    arr_new = numpy.copy(arr).astype(numpy.uint8)
    img = Image.fromarray(arr_new).convert('L')
    return numpy.asarray(img).astype(numpy.int32)


##################
# Sobel Kernels  #
##################

def sobel_x_derivative():
    """ Sobel X
    1 2 1
    2 4 2 * 1/16
    1 2 1
    """
    arr = (numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    return arr


def sobel_y_derivative():
    """ Sobel Y
    1 2 1
    2 4 2 * 1/16
    1 2 1
    """
    arr = (numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    return arr


##################
# Apply Kernels  #
##################

def single_sobel(img, kernel):
    (kernel_height, kernel_width) = kernel.shape
    (img_height, img_width) = img.shape

    img_out = img.copy()

    sobel_x = sobel_x_derivative()
    sobel_y = sobel_y_derivative()

    for i_row in range(1, img_height - 2):
        for i_col in range(1, img_width - 2):
            pixel_x = 0
            pixel_y = 0
            for k_row in range(0, kernel_height):  # Range is an open-ended interval!
                for k_col in range(0, kernel_width):
                    pixel_x += sobel_x[k_row][k_col] * img[i_row + -1 + k_row][i_col + -1 + k_col]
                    pixel_y += sobel_y[k_row][k_col] * img[i_row + -1 + k_row][i_col + -1 + k_col]

            img_out[i_row, i_col] = math.ceil(math.sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y)))
    return img_out


def apply_kernel(img, kernel):
    (kernel_height, kernel_width) = kernel.shape
    (img_height, img_width) = img.shape

    img_out = img.copy()

    for i_row in range(1, img_height - 2):
        for i_col in range(1, img_width - 2):
            pixel = 0
            for k_row in range(0, kernel_height):  # Range is an open-ended interval!
                for k_col in range(0, kernel_width):
                    offset_row = -1 + k_row
                    offset_col = -1 + k_col
                    img_val = img[i_row + offset_row][i_col + offset_col]
                    pixel += kernel[k_row][k_col] * img_val

            img_out[i_row, i_col] = pixel
    return img_out


def merge_sobel(img_x, img_y, expected):
    (img_height, img_width) = img_x.shape
    img_out = img_x.copy()

    for i_row in range(1, img_height - 2):
        for i_col in range(1, img_width - 2):
            ex = expected[i_row][i_col]
            pixel_x = img_x[i_row][i_col]
            pixel_y = img_y[i_row][i_col]
            val = math.ceil(math.sqrt((pixel_x * pixel_x) + (pixel_y * pixel_y)))

            img_out[i_row, i_col] = val
    return img_out


def main():
    #########
    # Input #
    #########

    # # Read in the image to an array.
    img_arr = image_to_array(sys.argv[1])

    # Make the image greyscale (1 byte per pixel)
    img_gray = convert_to_grayscale(img_arr)
    #
    # Compute the single-pass sobel filter.
    img_sobel = single_sobel(img_gray, sobel_x_derivative())
    #
    # Compute the multi-pass sobel filter.
    img_sobel_x = apply_kernel(img_gray, sobel_x_derivative())
    img_sobel_y = apply_kernel(img_gray, sobel_y_derivative())
    img_sobel_multi = merge_sobel(img_sobel_x, img_sobel_y, img_sobel)
    completedImg = Image.fromarray(img_sobel_multi.astype(numpy.uint8), 'L')
    completedImg.show()


start_time = time.time()
main()
print("Execution time: %s seconds" % (time.time() - start_time))