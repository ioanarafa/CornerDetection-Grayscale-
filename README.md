This project focuses on detecting corners in grayscale images using two well-known algorithms: Harris Corner Detection and Shi-Tomasi (Good Features to Track). The application is implemented in C++ using the OpenCV library and follows a standard computer vision pipeline: image loading, grayscale conversion, corner detection, and visualization.

The input image is read in color format and then converted to grayscale using cv::cvtColor, as corner detection relies on intensity gradients rather than color information. The Harris algorithm computes a corner response function using the eigenvalues of the autocorrelation matrix and highlights pixels with strong local intensity changes. The response is normalized and thresholded, and corners are visualized with black circles.

Shi-Tomasi improves upon Harris by selecting only the strongest corners based on the minimum eigenvalue of the gradient matrix. This method is particularly efficient for feature selection in applications like object tracking. Detected corners are displayed as green circles on the original image.

Both methods were tested on two different images (house.png and house2.png), and the results were visualized using cv::imshow. The project demonstrates the effectiveness of basic feature detection techniques and provides a good starting point for more advanced tasks in image analysis and computer vision.
