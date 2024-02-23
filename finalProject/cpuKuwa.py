from numba import cuda
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from PIL import Image

def HSVconversionCPU(rgb):
  shape = np.shape(rgb)
  hsv = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      R = rgb[i, j, 0]/255
      G = rgb[i, j, 1]/255
      B = rgb[i, j, 2]/255

      Max = max(R,G,B)

      hsv[i, j, 2] = Max
  
  return hsv

def addpaddingcpu(input, omega):
  shape = np.shape(input)
  output = np.zeros((shape[0]+omega*2,shape[1]+omega*2,3))
  
  for i in range(shape[0]):
    for j in range(shape[1]):
      output[(i+omega), (j+omega), 0] = input[i,j,0]
      output[(i+omega), (j+omega), 1] = input[i,j,1]
      output[(i+omega), (j+omega), 2] = input[i,j,2]

  return output


def generatewindowcpu(hsvinput, omega):
  shape = np.shape(hsvinput)
  output = np.zeros((shape[0],shape[1],4,(omega+1)**2))

  for i in range(shape[0]):
    for j in range(shape[1]):
      if (i < omega+1) or (j < omega+1) or (i > shape[0]- (omega+1)) or (j > shape[1]-(omega+1)):
        None
      else:
        count = 0
        for wx in range(omega+1):
          for wy in range(omega+1):
            output[i,j,0,count] = hsvinput[i-wx,j-wy,2]
            output[i,j,1,count] = hsvinput[i+wx,j-wy,2]
            output[i,j,2,count] = hsvinput[i-wx,j+wy,2]
            output[i,j,3,count] = hsvinput[i+wx,j+wy,2]

            count += 1
  
  return output


def calstd(input, omega):
  shape = np.shape(input)
  output = np.zeros((shape[0]+omega*2,shape[1]+omega*2,1))
  
  size = (omega+1)**2

  for i in range(shape[0]):
    for j in range(shape[1]):
      if (i < omega+1) or (j < omega+1) or (i > shape[0]- (omega+1)) or (j > shape[1]-(omega+1)):
        None
      else:
        std = np.zeros(4)
         
        for k in range(4):
          std[k] = np.std(input[i,j,k])
        
        minstd = min(std)
      
        for l in range(4):
          if std[l] == minstd:
            output[i,j,0] = l

  return output

def kuwaharacpu(input, stdwindow, omega):
  shape = np.shape(input)
  output = np.zeros(shape,dtype=np.uint8)
  size = (omega+1)**2

  for i in range(shape[0]):
    for j in range(shape[1]):
      if (i < omega+1) or (j < omega+1) or (i > shape[0]- (omega+1)) or (j > shape[1]-(omega+1)):
        None
      else:
        Rtotal = 0
        Gtotal = 0
        Btotal = 0
        for wx in range(omega+1):
          for wy in range(omega+1):
              if stdwindow[i, j, 0]==0:
                Rtotal += input[i-wx,j-wy,0]
                Gtotal += input[i-wx,j-wy,1]
                Btotal += input[i-wx,j-wy,2]

                output[i, j, 0] = Rtotal/size
                output[i, j, 1] = Gtotal/size
                output[i, j, 2] = Btotal/size

              elif stdwindow[i, j, 0]==1:
                Rtotal += input[i+wx,j-wy,0]
                Gtotal += input[i+wx,j-wy,1]
                Btotal += input[i+wx,j-wy,2]

                output[i, j, 0] = Rtotal/size
                output[i, j, 1] = Gtotal/size
                output[i, j, 2] = Btotal/size

              elif stdwindow[i, j,0]==2:
                Rtotal += input[i-wx,j+wy,0]
                Gtotal += input[i-wx,j+wy,1]
                Btotal += input[i-wx,j+wy,2]

                output[i, j, 0] = Rtotal/size
                output[i, j, 1] = Gtotal/size
                output[i, j, 2] = Btotal/size

              elif stdwindow[i, j,0]==3:
                Rtotal += input[i+wx,j+wy,0]
                Gtotal += input[i+wx,j+wy,1]
                Btotal += input[i+wx,j+wy,2]

                output[i, j, 0] = Rtotal/size
                output[i, j, 1] = Gtotal/size
                output[i, j, 2] = Btotal/size
  
  return output


im = plt.imread("./why-are-we-6010201214.jpg")
omega = 4

t1 = time.time()

hsv = HSVconversionCPU(im)
hsv2 = addpaddingcpu(hsv, omega)
windows = generatewindowcpu(hsv2, omega)
stdwindow = calstd(windows, omega)
imagepadded = addpaddingcpu(im, omega)
outputimage = kuwaharacpu(imagepadded, stdwindow, omega)
 
t2 = time.time()
print(t2-t1)


out = Image.fromarray(outputimage)
out.save("./kuwaharafilterCPU.jpeg")
 