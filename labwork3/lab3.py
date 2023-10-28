from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
import math
from numba import cuda

cuda.detect()

image = plt.imread("/home/dumdum/m2_subjects/advancedhpc2023/labwork3/example.jpg")

height, width = image.shape[0], image.shape[1]
pixelCount = width * height
blockSize = 64
gridSize = math.ceil(pixelCount) / blockSize
reshaped_img = np.reshape(image, (height * width, 3))

def grayified(img):
    r = image[:,:,0]*0.2989
    g = image[:,:,1]*0.5870
    b = image[:,:,2]*0.1140
    gray = r+g+b
    return gray

#CPU time

start =time.time()
result = grayified(image)
print("CPU time : ",abs(time.time()-start))

plt.imsave('test.png', image)
plt.imsave('cpu.png', result, cmap = 'gray')

devOutput = cuda.device_array((pixelCount), np.float64)
devInput = cuda.to_device(reshaped_img)
gridSize1 = math.ceil(gridSize)


@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = (src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3
    dst[tidx] = g

#GPU time
grayscale[gridSize1, blockSize](devInput, devOutput)
print("GPU time : ", abs(time.time()-start))

hostOutput = devOutput.copy_to_host()
hostOutput = np.reshape(hostOutput, (height, width))

plt.imsave('gpu.png',hostOutput, cmap = 'gray')
timer = []

#Block sizes
blockSizes = [16, 32, 64, 128, 256, 512, 1024]

for i in blockSizes:
    gridsize = pixelCount/i
    gridsize1 = math.ceil(gridSize)
    start = time.time()
    grayscale[gridsize1, i](devInput, devOutput)
    stop = time.time()
    timer.append(abs(start-stop))
    
gputime = timer[2]
print("GPU time : ", gputime)

fig, ax = plt.subplots()
ax.plot(blockSizes, timer)
ax.set_xlabel('Block size')
ax.set_ylabel('Compute time')
plt.savefig('Execution over block size.png')