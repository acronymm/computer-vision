import cv2
import numpy as np


def filter2D():
    filename = 'images/fruits.jpg'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # high-pass filter
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    result = cv2.filter2D(img, -1, kernel)
    cv2.imshow('image', img)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny():
    filename = 'images/lena.jpg'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    result = cv2.Canny(img, 50, 150)
    cv2.imshow('image', img)
    cv2.imshow('Canny', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def erodeNdilate():
    filename = 'images/char.jpg'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[0:2]
    kernel = np.ones((3, 3), np.uint8)
    result1 = cv2.erode(img, kernel, iterations=1)
    result2 = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow('image', img)
    cv2.imshow('erosion', result1)
    cv2.imshow('dilation', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morphologyEx():
    filename = 'images/char.jpg'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[0:2]
    kernel = np.ones((3, 3), np.uint8)
    result1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    result2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    result3 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('image', img)
    cv2.imshow('opening', result1)
    cv2.imshow('closing', result2)
    cv2.imshow('gradient', result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def applyFilter(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # high-pass filter
    highPass = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1],
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
    lowPassOnes = np.full((5, 5), 1)

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
    erodeNdilate()
    # morphologyEx()
    # canny()
    # filter2D()
    # filename = 'images/lena.jpg'
    # applyFilter(filename)
