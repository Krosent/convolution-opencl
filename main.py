# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/17815687/image-processing-implementing-sobel-filter
import numpy
import sys
import math
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

# Create a list of all the platform IDs
platforms = cl.get_platforms()

kernel = open("kernel.cl").read()
img = sys.argv[1]

vector_size = 1024

#context = cl.create_some_context()

# Step 1: Create a context.
context = cl.create_some_context()

# Create a queue to the device.
queue = cl.CommandQueue(context)

# Create the program.
program = cl.Program(context, kernel).build()

h_source = numpy.array(image_to_array(sys.argv[1]))
h_kernel = numpy.array(kernel)

# Create the result vector.
h_output = numpy.empty(size).astype(numpy.float32)

d_source = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_source)
d_kernel = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_kernel)

# Create the memory on the device to put the result into.
d_output = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_output.nbytes)

# Create two vectors to be added.
#h_a = numpy.random.rand(vector_size).astype(numpy.float32)
#h_b = numpy.random.rand(vector_size).astype(numpy.float32)
#h_c = numpy.empty(vector_size).astype(numpy.float32)
#h_d = numpy.empty(vector_size).astype(numpy.float32)

# Create the result vector.
#h_x = numpy.empty(vector_size).astype(numpy.float32)
#h_y = numpy.empty(vector_size).astype(numpy.float32)
#h_z = numpy.empty(vector_size).astype(numpy.float32)

#d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
#d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
#d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)
#d_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_d)

# Create the memory on the device to put the result into.
#d_x = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_x.nbytes)
#d_y = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_y.nbytes)
#d_z = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_z.nbytes)

#vadd = program.vadd

#vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

#vadd(queue, h_a.shape, None, d_a, d_b, d_x, vector_size)

#vadd(queue, h_a.shape, None, d_x, d_c, d_y, vector_size)

#vadd(queue, h_a.shape, None, d_y, d_d, d_z, vector_size)

# Wait for the queue to be completely processed.

#queue.finish()

# Read the array from the device.
# cl.enqueue_copy(queue, h_z, d_z)

##################
# Apply Kernels  #
##################

def single_sobel(img, kernel):
    (img_height, img_width)  = img.shape
    (kernel_height, kernel_width) = kernel.shape

    image_size = img_height * img_width
   
    #h_d = numpy.empty(vector_size).astype(numpy.float32)
    h_source = numpy.array(img)
    h_kernel = numpy.array(kernel)

    # Create the result vector.
    #h_output = numpy.empty(matrix_size).astype(numpy.float32)
    h_output = numpy.array(img)

    d_source = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_source)
    d_kernel = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_kernel)

    # Create the memory on the device to put the result into.
    d_output = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_output.nbytes)
    
    img_out = img.copy()
    

    sobel_x = sobel_x_derivative()
    sobel_y = sobel_y_derivative()

    #### Test
    #print("----")
    #print(h_output)
    #print("----")
    #### -----

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


#########
# Input #
#########

#print("HEY")



# # Read in the image to an array.
img_arr = image_to_array(sys.argv[1])

# Make the image greyscale (1 byte per pixel)
img_gray = convert_to_grayscale(img_arr)
#
# Compute the single-pass sobel filter.
img_sobel = single_sobel(img_gray, sobel_x_derivative())
#
"""
# Compute the multi-pass sobel filter.
img_sobel_x = apply_kernel(img_gray, sobel_x_derivative())
img_sobel_y = apply_kernel(img_gray, sobel_y_derivative())
img_sobel_multi = merge_sobel(img_sobel_x, img_sobel_y, img_sobel)
save_image(img_sobel, "output.png", "L")

(h, w) = img_sobel.shape
for r in range(0, h):
    for c in range(0, w):
        if img_sobel[r][c] != img_sobel_multi[r][c]:
            print("[{}][{}] differ!".format(r, c))

"""