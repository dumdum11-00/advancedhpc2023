from numba import cuda
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from PIL import Image


im = plt.imread("./why-are-we-6010201214.jpg")

shape = np.shape(im)
shape

@cuda.jit
def HSVconversion(rgb, hsv):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

  R = rgb[tidx, tidy, 0]/255
  G = rgb[tidx, tidy, 1]/255
  B = rgb[tidx, tidy, 2]/255

  Max = max(R,G,B)
  Min = min(R,G,B)

  Delta = Max - Min

  if Delta ==0:
    H = 0
  elif Max == R:
    H = 60*((((G-B)/Delta))%6)
  elif Max == G:
    H = 60*(((B-R)/Delta)+2)
  elif Max == B:
    H = 60*(((B-R)/Delta)+6)

  if Max == 0:
    S = 0
  else:
    S = Delta / Max

  V = Max

  hsv[tidx, tidy, 0] = H
  hsv[tidx, tidy, 1] = S
  hsv[tidx, tidy, 2] = V

@cuda.jit
def addpadding(input, output, omega):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

  output[(tidx+omega), (tidy+omega), 0] = input[tidx,tidy,0]
  output[(tidx+omega), (tidy+omega), 1] = input[tidx,tidy,1]
  output[(tidx+omega), (tidy+omega), 2] = input[tidx,tidy,2]

@cuda.jit
def generatewindow(hsvinput, omega, shape, output):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  
  if (tidx < omega) or (tidy < omega) or (tidx > shape[0]+(omega)) or (tidy > shape[1]+(omega)):
    return

  count = 0
  for wx in range(omega+1):
    for wy in range(omega+1):
      w1 = hsvinput[tidx-wx,tidy-wy,2]
      w2 = hsvinput[tidx+wx,tidy-wy,2]
      w3 = hsvinput[tidx-wx,tidy+wy,2]
      w4 = hsvinput[tidx+wx,tidy+wy,2]
       
      output[tidx,tidy,0,count] = w1 
      output[tidx,tidy,1,count] = w2
      output[tidx,tidy,2,count] = w3
      output[tidx,tidy,3,count] = w4
       
      count += 1

@cuda.jit
def calstd(input, omega, output):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  size = (omega+1)**2

  if (tidx < omega) or (tidy < omega) or (tidx > shape[0]+omega) or (tidy > shape[1]+omega):
    return

  w0 = input[tidx,tidy,0]
  w1 = input[tidx,tidy,1]
  w2 = input[tidx,tidy,2]
  w3 = input[tidx,tidy,3]

  #Window 0
  t = 0
  for i in w0:
    t = t + i
  mean0 = t/size

  t2 = 0 
  for i in w0:
    t2 = t2 + (i- mean0)**2
  
  std0 = math.sqrt(t2/size)

  #Window 1
  t = 0
  for i in w1:
    t = t + i
  mean1 = t/size

  t2 = 0 
  for i in w1:
    t2 = t2 + (i- mean1)**2

  std1 = math.sqrt(t2/size)

  #Window 2
  t = 0
  for i in w2:
    t = t + i
  mean2 = t/size

  t2 = 0 
  for i in w2:
    t2 = t2 + (i- mean2)**2

  std2 = math.sqrt(t2/size)

  #Window 3
  t = 0
  for i in w3:
    t = t + i
  mean3 = t/size

  t2 = 0 
  for i in w3:
    t2 = t2 + (i- mean3)**2

  std3 = math.sqrt(t2/size)
  output[tidx, tidy, 0] = std1

  
  minstd = min(std0, std1, std2, std3)
  
  for i in (std0,std1,std2,std3):
    if i < minstd:
      minstd = i

  if minstd == std0:
    output[tidx, tidy, 0] = 0
  elif minstd == std1:
    output[tidx, tidy, 0] = 1
  elif minstd == std2:
    output[tidx, tidy, 0] = 2
  elif minstd == std3:
    output[tidx, tidy, 0] = 3

@cuda.jit
def kuwahara(input, stdwindow, omega, output):
  tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
  tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
  size = (omega+1)**2

  if (tidx < omega) or (tidy < omega) or (tidx > shape[0]+(omega)) or (tidy > shape[1]+(omega)):
   return

  Rtotal = 0
  Gtotal = 0
  Btotal = 0
  for wx in range(omega+1):
     for wy in range(omega+1):
        if stdwindow[tidx, tidy,0]==0:
          Rtotal += input[tidx-wx,tidy-wy,0]
          Gtotal += input[tidx-wx,tidy-wy,1]
          Btotal += input[tidx-wx,tidy-wy,2]

          output[tidx, tidy, 0] = Rtotal/size
          output[tidx, tidy, 1] = Gtotal/size
          output[tidx, tidy, 2] = Btotal/size
       
        elif stdwindow[tidx, tidy,0]==1:
          Rtotal += input[tidx+wx,tidy-wy,0]
          Gtotal += input[tidx+wx,tidy-wy,1]
          Btotal += input[tidx+wx,tidy-wy,2]

          output[tidx, tidy, 0] = Rtotal/size
          output[tidx, tidy, 1] = Gtotal/size
          output[tidx, tidy, 2] = Btotal/size        

        elif stdwindow[tidx, tidy,0]==2:
          Rtotal += input[tidx-wx,tidy+wy,0]
          Gtotal += input[tidx-wx,tidy+wy,1]
          Btotal += input[tidx-wx,tidy+wy,2]

          output[tidx, tidy, 0] = Rtotal/size
          output[tidx, tidy, 1] = Gtotal/size
          output[tidx, tidy, 2] = Btotal/size 

        elif stdwindow[tidx, tidy,0]==3:
          Rtotal += input[tidx+wx,tidy+wy,0]
          Gtotal += input[tidx+wx,tidy+wy,1]
          Btotal += input[tidx+wx,tidy+wy,2]

          output[tidx, tidy, 0] = Rtotal/size
          output[tidx, tidy, 1] = Gtotal/size
          output[tidx, tidy, 2] = Btotal/size

omega = 4
image = cuda.to_device(im)
blockSize = (8,8)
gridSize = (math.ceil(shape[0]/blockSize[0]),math.ceil(shape[1]/blockSize[1]))

t1 = time.time()

#calculate hsv
hsvdev1 = cuda.device_array((shape[0],shape[1],3), np.float64)
HSVconversion[gridSize, blockSize](image, hsvdev1)

#add padding to HSV
hsvdev2 = cuda.device_array((shape[0]+omega*2,shape[1]+omega*2,3), np.float64)
addpadding[gridSize, blockSize](hsvdev1, hsvdev2, omega)

#create an array of all points for each pixel
windowdev = cuda.device_array((shape[0]+omega*2,shape[1]+omega*2,4,(omega+1)**2), np.float64)
generatewindow[gridSize, blockSize](hsvdev2, omega, shape, windowdev)

#calculate std and point out the windows have lowest std
stdwindow = cuda.device_array((shape[0]+omega*2,shape[1]+omega*2,1), np.int8)
calstd[gridSize, blockSize](windowdev, omega, stdwindow)

#add padding to original image to prepare for kuwahara
impadded = cuda.device_array((shape[0]+omega*2,shape[1]+omega*2,3), np.uint8)
addpadding[gridSize, blockSize](image, impadded, omega)

#apply kuwahara filter
outputimage = cuda.device_array((shape[0]+omega*2,shape[1]+omega*2,3), np.uint8)
kuwahara[gridSize, blockSize](impadded, stdwindow, omega, outputimage)

t2 = time.time()

print(t2 - t1)

outputimagehost = outputimage.copy_to_host()
out = Image.fromarray(outputimagehost)
out

out.save("./kuwaharafilterGPU.jpeg")