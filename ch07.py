import cv2
import numpy as np


def applyFilter(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # high-pass filter
    highPass = np.array([
        [-1,-1,-1],
        [-1, 9,-1],
        [-1,-1,-1],
    ])
    
    # low-pass Gaussian filter
    lowPassGaussian = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 4, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.float32)

    lowPassGaussian = lowPassGaussian * 5
    lowPassGaussian = lowPassGaussian/lowPassGaussian.sum()

    # low-pass ones filter
    lowPassOnes = np.full((5,5), 1)

    lowPassOnes = lowPassGaussian/lowPassGaussian.sum()

    highPassResult = cv2.filter2D(image, -1, highPass)
    lowPassGaussianResult = cv2.filter2D(image, -1, lowPassGaussian)
    lowPassOnesResult = cv2.filter2D(image, -1, lowPassOnes)

    cv2.imshow('original', image)
    cv2.imshow('highPassResult', highPassResult)
    cv2.imshow('lowPassGaussianResult', lowPassGaussianResult)
    cv2.imshow('lowPassOnesResult', lowPassOnesResult)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = 'images/lena.jpg'
    applyFilter(filename)