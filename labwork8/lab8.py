from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
import math
from numba import cuda

block_size_list = [(2,2),
                   (4, 4),
                   (8, 8),
                   (16, 16),
                   (32, 32)]

def dual_tuple_division(x, y):
    return_tuple = []
    for i, ii in zip(x, y):
        return_tuple.append(math.ceil(ii/i))
    return tuple(return_tuple)


@cuda.jit
def rgb_hsv(src, dst):
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Prepare
    r, g, b = src[tx, ty, 0], src[tx, ty, 1], src[tx, ty, 2]

    if r == 255 and g == 255 and b ==255:
        r = 255
        g,b = 0, 0

    max_value = np.float64(max(r, g, b))
    min_value = np.float64(min(r, g, b))


    delta = np.float64(max_value - min_value)

    # Calculate H 
    if delta == np.uint64(0):
        H = np.uint64(0)
    elif max_value == r:
        H = np.float64(60* (((g-b)/delta)%6))
    elif max_value == g:
        H = np.float64(60* ((b-r)/delta+2))
    elif max_value == b:
        H = np.float64(60* ((r-g)/delta+4))
    
    # Calculate S
    if max_value == np.uint64(0):
        S = np.uint64(0)
    elif max_value != np.uint64(0):
        S = np.float64(delta/ max_value)

    V = max_value

    dst[tx, ty, 0], dst[tx, ty, 1], dst[tx, ty, 2]= H, S, V


@cuda.jit
def hsv_rgb(src, dst):
    tx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ty = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    H, S, V = src[tx, ty, 0], src[tx, ty, 1], src[tx, ty, 2]

    # Prepare
    d = H/60
    hi = np.uint8(d%6)
    f = d - hi
    l = V * (1-S)
    m = V * (1- f * S)
    n = V * (1 - (1 - f)* S)

    # Conversion
    if (0 <= H < 60):
        r, g, b = V, n, l
    elif (60 <= H < 120):
        r, g, b = m, V, l
    elif (120 <= H < 180):
        r, g, b = l, V, n
    elif (180 <= H < 240):
        r, g, b = l, m, V
    elif (240 <= H < 300):
        r, g, b = n, l, V
    elif (300 <= H < 360):
        r, g, b = V, l, m
        
    dst[tx, ty, 0], dst[tx, ty, 1], dst[tx, ty, 2]= r, g, b

for block_size in block_size_list:
    # Load and ignore alpha channel
    img = plt.imread("/home/dumdum/m2_subjects/advancedhpc2023/labwork3/example.jpg")[:, :, :3]
    img = np.float64(img)
    img /= 255
    img = np.ascontiguousarray(img)
    h, w, c = img.shape

    out = np.array(img, copy=True)
    out2 = np.zeros(img.shape)

    grid_size = dual_tuple_division(block_size, (h, w))
    start = time.time()

    A = cuda.to_device(img)
    B = cuda.to_device(out)
    C = cuda.to_device(out2)
    rgb_hsv[grid_size, block_size](A, B)
    hsv_rgb[grid_size, block_size](B, C)

    plt.imsave('output-lab8.png',C.copy_to_host())
