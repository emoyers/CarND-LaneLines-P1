#importing some useful packages
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)  

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_r = line_l = np.array([np.zeros(4)])
    count_r = count_l = 0
    
    #getting 2 average lines one for the right lines and one for the left ones 
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if slope > 0:
            	if count_r == 0:
            		line_r = line
            	else:
            	    line_r = (line_r + line) / 2
            	count_r += 1
            else:
            	if count_l == 0:
            		line_l = line
            	else:
            	    line_l = (line_l + line) / 2
            	count_l += 1

    #getting slope(m) and b for each line y = mx + b
    for x1,y1,x2,y2 in line_r:
    	slope_r = ((y2-y1)/(x2-x1))
    	b_r = y1 - (slope_r*x1)
    #calculating the point that intersect the X axis
    y2_r = img.shape[0]
    x2_r = np.int16((y2_r-b_r) / slope_r)

    for x1,y1,x2,y2 in line_l:
    	slope_l = ((y2-y1)/(x2-x1))
    	b_l = y1 - (slope_l*x1)
    #calculating the point that intersect the X axis
    y1_l = img.shape[0]
    x1_l = np.int16((y1_l-b_l) / slope_l)

    #calculating the point where line_l and line_r intersect
    #x_top = np.int16((b_r - b_l) / (slope_l - slope_r))
    #y_top = np.int16((slope_r * x_top) + b_r)
    y_top = np.int16(img.shape[0]*0.6)
    x1_r = np.int16((y_top-b_r) / slope_r)
    x2_l = np.int16((y_top-b_l) / slope_l)

    #Printing images in original img cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img, (x1_r, y_top), (x2_r, y2_r), color, thickness)
    cv2.line(img, (x2_l, y_top), (x1_l, y1_l), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
    
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(path):
	#reading in an image
	if type(path) == type("String"):
		image = mpimg.imread(path)
	else:
		image = path

	#making the image gray
	gray_image = grayscale(image)

	#blur image to remove noise and make easier to fine edges
	blur_image = gaussian_blur(gray_image, 7)

	#get edges of the blur image to identify easily the lines in the road
	edges = canny(blur_image, 145, 170)

	#eliminate the rest of the pixels that al not within the poligon with vertices [150,540],[400,300],[600,300],[900,540]
	vertices = np.array([[np.int32(image.shape[1]*0.10),image.shape[0]],[(image.shape[1]*0.39),(image.shape[0]*0.65)],[(image.shape[1]*0.57),(image.shape[0]*0.57)],[(image.shape[1]*0.94),image.shape[0]],[(image.shape[1]*0.45),(image.shape[0]*0.87)],[(image.shape[1]*0.49),(image.shape[0]*0.87)],], np.int32)
	edges_w_region = region_of_interest(edges, [vertices])

	#Apply Hough Transform to convert points into lines
	RHO = 2
	THETA = np.pi/180
	THRESHOLD = 15
	MIN_LINE_LENGTH = 40
	MAX_LINE_GAP = 20
	hough_image = hough_lines(edges_w_region, RHO, THETA, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)

	#
	alpha = 0.60
	beta = ( 1.0 - alpha );
	init_image_w_lines = weighted_img(hough_image, image,alpha,beta,0.)

	#printing out some stats and plotting
	print('This image is:', type(init_image_w_lines), 'with dimensions:', init_image_w_lines.shape)
	fig = plt.figure(figsize=(20, 20))
	fig.add_subplot(2, 2, 1)
	plt.imshow(image) 
	fig.add_subplot(2, 2, 2)
	plt.imshow(edges_w_region, cmap= 'gray')
	fig.add_subplot(2, 2, 3)
	plt.imshow(hough_image, cmap= 'gray') 
	fig.add_subplot(2, 2, 4)
	cv2.imwrite(os.path.join("test_images_output/" , "out_"+file), cv2.cvtColor(init_image_w_lines, cv2.COLOR_RGB2BGR))
	plt.imshow(init_image_w_lines) 
	# if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
	#plt.show()
	return init_image_w_lines

list_images = os.listdir("test_images/")
for file in list_images:
   path_image = "test_images/"+file
   print ("test_images/"+file)
   process_image(path_image)

white_output = 'test_videos/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip(white_output).subclip(0,5)
clip1 = VideoFileClip(white_output)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile("test_videos_output/final_video.mp4")