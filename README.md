# aerial-image-stitching

![result](/img/result.png)
A simple implementation that stiches images from UAV view into a map. First estimate camera movement by applying SIFT feature matching and ratio test to expand the current image, then warp the new frame to the current frame using the estimated homography matrix.

## Dependencies
* Numpy
* matplotlib
* OpenCV 3.4.2 (For image loading and SIFT feature extraction)
* scikit-learn 0.22.1

## Data source
Example images are from [Sensefly dataset](https://www.sensefly.com/education/datasets/?dataset=5580&sensors%5B%5D=24).
