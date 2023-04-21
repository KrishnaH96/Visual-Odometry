# Camera Pose Estimation using Homography

## Problem 1.1: Image Processing Pipeline for Paper Corner Detection
- Read each frame of the video using OpenCV `Video Capture` object.
- Resize each frame using a scale factor of 0.5 in both x and y direction.
- Convert the BGR frame to grayscale frame.
- Apply a threshold to the grayscale image to create a binary mask.
- Use the histogram of the grayscale image intensity to choose threshold values.
- Use closing action of morphology operation to fill the gaps in dominant edges and remove small items.
- Use a 5x5 kernel for the closing operation.
- Detect the edges using Canny algorithm.
- Use the Hough transform to detect the strong edges.
- Create an accumulator array that contains bins for each possible line that can be drawn in the input image.
- Each bin represents a line defined by a distance and angle from the origin of the image.
- Iterate through every edge pixel to calculate the polar coordinates of the edge (perpendicular distance from the center(d) and inclination with position x axis(theta)).
- Increment the corresponding accumulator position if the pair of same (d, theta) is identified during the iterations.
- Once we iterate over every pixel and the accumulator is iterated for all the theta values for each pixel, the (d, theta) pairs with the highest number of votes are extracted. These pairs represent the dominant lines in the image, in our case the edges of the paper.
- Change these polar coordinates to give the lines in the original frame.
- Draw these lines that will match the edges of the paper.
- Use the slope-intercept forms of these four lines to compute the intersection of pair of these lines to get four corners.

## Problem 1.2 & 1.3: Homography Computation and Decomposition
- Map the detected corners to match the page coordinates that are defined by assigning the origin to the top left corner and using the height and width of the paper.
- Construct matrix A with 9 columns and 8 rows using the corresponding points from the two frames.
- Use SVD to solve A to get the homography matrix.
- Define and scale down the intrinsic camera matrix by a scaling factor.
- Using the Homography matrix and intrinsic matrix, compute the rotation and translation matrix for the movement of camera in the video.
- For the translation matrix, extract the zeroth, first and second element from the translation matrix to get the travel of camera in X,Y and Z direction.
- For rotation, use formulas to get the roll, pitch and yaw from the rotation matrix.
- Create a 3D plot to show the movement of camera using X,Y and Z coordinates from each frame.
- Plot the roll, pitch and yaw with respective frame numbers as X axis.
