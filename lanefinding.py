import numpy as np
import cv2

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(2.5/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def averageLine(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg = np.average(left_fit, axis = 0)
    right_fit_avg = np.average(right_fit, axis = 0)
    # Exception Handling if parameters not caught in a frame
    try:
        left_line = make_coordinates(image, left_fit_avg)
        right_line = make_coordinates(image, right_fit_avg)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e, '\n') # print error to console
        return None

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # Read in and grayscale the image
    kernel_size = 5 # Define a kernel size
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) # Apply Gaussian smoothing
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return edges

def displayLines(image, lines):
    line_image = np.zeros_like(image) # creating a blank to draw lines on
    # Iterate over the output "lines" and draw lines on a blank image
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    return line_image

def regionofInterest(image):
    ignore_mask_color = 255
    # This time we are defining a triangle to mask
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    # Next we'll create a masked canny_image image using cv2.fillPoly()
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, ignore_mask_color)
    masked_edges = cv2.bitwise_and(image, mask)
    return masked_edges

def houghTransform(masked_edges):
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 100    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    return lines

def plotImage(canny_image, lane_image, line_image):
    # Create a "color" binary image to combine with line image
    # color_edges = np.dstack((canny_image, canny_image, canny_image))
    # Draw the lines on the edge image
    color_edges = np.dstack((canny_image, canny_image, canny_image))
    combo_image1 = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    combo_image2 = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)
    cv2.imshow("result",combo_image2)

def findLane(image):
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    masked_edges = regionofInterest(canny_image)
    lines = houghTransform(masked_edges)
    averaged_lines = averageLine(lane_image, lines)
    line_image = displayLines(lane_image,averaged_lines)
    plotImage(canny_image, lane_image, line_image)

# MAIN()
# image = cv2.imread('test_image.jpg')

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    findLane(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
