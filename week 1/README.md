# Training Documentation Week 1

## Nov 3 ( Day 1)

### Learned :

- Creating a Blank image  
- Splitting into Multi-channels  
- Color Image converting ( HSV, RGB, LAB, GRAY)  
- Add the pixels of different image or channel.  
- Slicing the image using array slicing.  
- Color the specify region.  
- Change the resize of the image using the cv.resize with lossy and lossless conventions.  
- Rotating the image by array method by shifting the rows and columns.  


## Nov 4 ( Day 2)

### Learned :

Different Thresholding Techniques like Simple , Adaptive, Ostu Thershold.  

In simple threshold,  

THRESH_BINARY ==> 0 IF pix < threshold ELSE 255.  

THERSH_BINARY_INV is reciprocal for binary threshold.  

THRESH_TRUNC, thresh IF pix > thresh ELSE pix  

THRESH_TOZERO, pix IF pix>thresh ELSE 0  

THRESH_TOZERO_INV, 0 IF pix > thresh ELSE pix  

In Adaptive Thresholding, it split the image pixel into small block (11x11) pixel and find the local threshold, then it do the thresholding.  

ADAPTIVE_THRESH_MEAN_C  ==> it take mean of small block for find thershold and subtract with the constant ( C)  

ADAPTIVE_THRESH_GAUSSIAN_C  ==> it find threshold by gaussian kernel.  

In Ostu Thresholding, It histogram the count of pixel value from 0 – 255 and find the threshold that split the 2 peak into 2 class, foreground and background.  

Based on the threshold, it do binary_thresholding.  

### Edge Detection :

**Sobel Edge Detection:**  
It works to 2 directions with 2 kernels [[-1,-2,-1],[0,0,0],[1,2,1]] , [[-1,0,1],[-2,0,2],[-1,0,1]] for both x and y to find the vertical and horizonal edges.  
Then find the magnitude of the two images (sobelx,sobely) using cv.magnitude or cv.addweighted(), Pythagoras formula.  

**Scharr Edge Detection:**  
It work similar to sobel but the kernel is different.  
Kernel like [[-3,-10,-3],[0,0,0],[3,-10,3]]  

**Canny Edge Detection:**  
Canny is the basic edge detection for all types of unshaped boundary edges.  
Use Gaussian Blur and sobel to find the 1 derivation, then use Non-Maximum Suppression and connect the weak edge with strong edge.  
Non- Maximum Suppression is checking the neighbor pixel to reduce the thick edge to thin edge.  
Atlast check for the weak edge (between the max thresh and min thresh) to connect with strong edge means it considers strong edge.  


## Nov 5 ( Day 3)

### Learned :

**Smoothing Algorithm.**

Gaussian Blur : it work like convoluting the gaussian kernel 1/16 [[1,2,1],[2,4,2],[1,2,]] with the image.  It reduce the gaussian nosie and smooth the edge of object.  

Average Blur : It take the kernel size , make it average and store the average value in the center pixel.  

Median Blur : it also take the kernel size pixel array from the image , find the median and store in center pixel.  

Bilaternal Blur: It finds the spital and intensity differences. If the difference is higher means the pix is edge of object, else it negative means, it does not edge and set the value minus fined differences.  

### Gaussian Based Edge Detection:

**Differences of Gaussian:**  
It works by subtracting the gaussian kernel result with maximum sigma with the minimun sigma.  
If the result is postive means it is an edge, else if negative means it not edge may nosie remove it as 0.  

**Laplacian of Gaussian:**  
It has 2 steps, one is gaussian blur, and another one is Laplacian kernel to find the value is cross the zero or not.  
Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]. If the value is positive means it in dark side, else if value in negative side means it in the positive side pixel else the value is 0 means the edge of object in image.  


## Nov 6 ( Day 4)

### Learned :

**Contour Detection and Shape Analysis**

Contours are the boundaries that connect continuous points along the same intensity or color.  

cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  

CHAIN_APPROX_NONE => stores all the contour points.  
CHAIN_APPROX_SIMPLE => compresses and keeps only the corner points, reducing memory.  

cv.drawContours(image, contours, -1, (0,255,0), 2)  

### Contour Properties:

Area: cv.contourArea(cnt)  
Perimeter / Arc Length: cv.arcLength(cnt, True)  
Approximation: cv.approxPolyDP(cnt, epsilon, True)  

epsilon is a factor (like 0.02 × arcLength).  
Used to detect geometric shapes such as triangle, rectangle, square, pentagon, etc.  

### Shape Detection

Based on the number of approximated corner points:  

3 points → Triangle  
4 points → Square or Rectangle (based on aspect ratio)  
5 points → Pentagon  
6 points → Circle or Ellipse  

Aspect Ratio:  
aspect_ratio = w / h  
If aspect ratio ≈ 1 => Square, else Rectangle.  

### Contour Hierarchy:

Each contour can have a parent or child relationship.  
Hierarchy provides the structural relationship between contours:  
cv.RETR_TREE retrieves all contours and organizes them in a hierarchy (parent-child).  
cv.RETR_LIST retrieves all contours without any hierarchy information.  

### Moments:

M = cv.moments(cnt)  
Used to find the centroid (center of mass) of a contour:  
cx = int(M['m10']/M['m00'])  
cy = int(M['m01']/M['m00'])  

### Bounding Shapes:

Bounding Rectangle: x,y,w,h = cv.boundingRect(cnt)  
Minimum Enclosing Circle: cv.minEnclosingCircle(cnt)  
Convex Hull: checks whether a contour is convex or concave.  


## Week 1 Conclusion — Image Fundamentals & Edges

- Understood the basics of images and color spaces — Learned how an image is made of pixels, and how to work with RGB, HSV, LAB, and Grayscale formats for different visual processing needs.  
- Explored edge detection techniques — Learned Sobel, Scharr, and Canny methods to detect object boundaries and understand how gradients reveal shape outlines.  
- Applied image smoothing and filtering — Used Gaussian, Median, and Bilateral filters to remove noise and make edges cleaner for better detection results.  
- Learned contour detection and shape analysis — Found object boundaries, calculated area, perimeter, and recognized basic shapes using contours and hierarchy.  
- Built a simple shape detector — Combined all techniques (thresholding, edges, contours) to identify and label shapes like triangles, rectangles, and circles.  
