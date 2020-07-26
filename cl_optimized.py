# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
import numpy
import sys
import math
import time
import pyopencl as cl
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
# OPENCL         #
##################
def main():
    # Create a list of all the platform IDs
    platforms = cl.get_platforms()

    kernel = open("kernel_opt.cl").read()
    img = Image.open(sys.argv[1])
    img_arr = numpy.asarray(img).astype(numpy.uint8)

    img_gray = convert_to_grayscale(img_arr)

    (height, width, channels) = img_arr.shape
    img_src = img_gray.reshape((height * width))
    (kernel_height, _kernel_weight) = sobel_x_derivative().shape
    kernel_n = kernel_height

    #print("-- Iamge Size --")
    image_size = height * width
    #print(image_size)
    #print("--- ---")

    # Step 1: Create a context.
    context = cl.create_some_context()

    # Create a queue to the device.
    queue = cl.CommandQueue(context)

    # Create the program.
    program = cl.Program(context, kernel).build()

    h_source = img_src
    h_sobel_x = sobel_x_derivative().astype(numpy.int32)
    h_sobel_y = sobel_y_derivative().astype(numpy.int32)

    h_out = numpy.empty(image_size).astype(numpy.int32)

    d_src = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_source)
    d_sobel_x = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_sobel_x)
    d_sobel_y = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_sobel_y)

    # Create the memory on the device to put the result into.
    d_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_out.nbytes)

    sobel = program.sobel

    sobel.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None, None])

    sobel(queue, (height, width), None, height, width, kernel_n, d_src, d_sobel_x, d_sobel_y, d_out)

    queue.finish()

    cl.enqueue_copy(queue, h_out, d_out)

    # Let us reshape
    reshaped = h_out.reshape((height, width))

    #save_image(h_out, "output_cl.png", "L")

    #OLD
    #completedImg = Image.fromarray(reshaped, 'L')
    #NEW 
    completedImg = Image.fromarray(reshaped.astype(numpy.uint8), 'L')
    #save_image(reshaped, "output_cl.png", "L")
    completedImg.show()
    #save_image(completedImg, "output_cl.png")

    
start_time = time.time()
main()
print("Execution time: %s seconds" % (time.time() - start_time))