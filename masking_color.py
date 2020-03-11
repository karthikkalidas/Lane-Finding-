import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read image and print out stats
image = mpimg.imread("test.jpg")

# Grab x and y size to make copy of image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image) # Pixel color threshold
line_image = np.copy(image) # After threshold and masking
print("xsize is ",xsize,"ysize is ",ysize)

# Color threshold definition
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold,blue_threshold,green_threshold]

# Triangle Mask
left_bottom = [0,540]
right_bottom = [960,540]
apex = [450,300]

# Fitting lines (y=mx+c) through the triangle points
fit_left = np.polyfit((left_bottom[0],apex[0]),(left_bottom[1],apex[1]),1)
fit_right = np.polyfit((right_bottom[0],apex[0]),(right_bottom[1],apex[1]),1)
fit_bottom = np.polyfit((left_bottom[0],right_bottom[0]),(left_bottom[1],right_bottom[1]),1)

# Identify pixels below threshold
color_thresholds = (image[:,:,0]<rgb_threshold[0]) | \
                   (image[:,:,1]<rgb_threshold[1]) | \
                   (image[:,:,2]<rgb_threshold[2])

# Find region inside the lines
XX, YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY > (XX*fit_left[0]+fit_left[1])) & \
                    (YY > (XX*fit_right[0]+fit_right[1])) & \
                    (YY < (XX*fit_bottom[0]+fit_bottom[1]))

# Mask color selection
color_select[color_thresholds] = [0, 0, 0]
# Find where image is both colored correctly and within the region of interest
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Show image
plt.imshow(line_image)
x = [left_bottom[0],right_bottom[0],apex[0],left_bottom[0]]
y = [left_bottom[1],right_bottom[1],apex[1],left_bottom[1]]
plt.plot(x,y,'b--',lw=4)
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()
mpimg.imsave("test-after-masking.png", color_select)
