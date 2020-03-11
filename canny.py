import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('exit-ramp.jpg')

# Grayscale conversion
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.imshow(gray,cmap = 'gray')

# Thresholds for edge detection
low_threshold = 50
high_threshold = 150

# Gaussian Filer
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)

# Canny edge detection
edges = cv2.Canny(gray, low_threshold, high_threshold)
plt.imshow(edges,cmap = 'Greys_r')

plt.show()
