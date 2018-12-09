# **Finding Lane Lines on the Road** 

## Emilio Moyers Reflection


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/out_gray_solidWhiteCurve.jpg "Grayscale"
[image2]: ./test_images_output/out_blur_solidWhiteCurve.jpg "Grayscale"
[image3]: ./test_images_output/out_edges_solidWhiteCurve.jpg 
[image4]: ./test_images_output/out_edges_w_region_solidWhiteCurve.jpg
[image5]: ./test_images_output/out_hough_image_solidWhiteCurve.jpg
[image6]: ./test_images_output/out_solidWhiteCurve.jpg

---

### Reflection

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I blured the image to clean the image for excessive noise which make simple to find edges. After that used Canny algorith to find the edges
of the images base on the gradients each part of the image has. Next, I remove the part of the image that I was not interested in with the function fillPolly, giving it the points of the polygon I'm interested on. Then I apply the Hough Transform to convert the points found with Canny algorith into lines, giving the maximum reparation between lines, a threshold and the minimum lenght of the line. Finally I averaged and extropolate the lines to be draw in the original image or video.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by sepation the lines in left and
right base on their slopes to take an average 2 lines, one for the right side and one for the left side. After that I calculate the m and b of each line (y = mx + b) to be able to calculate the points of each line intersecting x axis and intersection x equal to 0.6 the maximum value of y.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]
**Gray Image**

![alt text][image2]
**Blur Image**

![alt text][image3]
**Edges of the image using Canny algorithm**

![alt text][image4]
**Egdes of the region of interest**

![alt text][image5]
**Hough transform with averaged and extrapolated lines**

![alt text][image6]
**Original image with lines**


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when a line is present in the ground without being a lane line. 

Another shortcoming could be the diferent types of light for example shadows or night light.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to refine the area of interested to maintain a more constant line without errors.

Another potential improvement could be to make a better filtering to avoid confusion with shadows or other lines in the paviment.
