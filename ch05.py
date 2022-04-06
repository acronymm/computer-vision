import cv2
from cv2 import reduce
import numpy as np

def bitwise_overlap():
    image = cv2.imread("images/bit_test.jpg", cv2.IMREAD_COLOR)
    logo = cv2.imread("images/logo.jpg", cv2.IMREAD_COLOR)
    if image is None or logo is None: raise Exception("File error")

    masks = cv2.threshold(logo, 220, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("masks before splitting", masks)
    masks = cv2.split(masks)

    cv2.imshow("image", image)
    cv2.imshow("logo", logo)
    cv2.imshow("mask-blue", masks[0])
    cv2.imshow("mask-green", masks[1])
    cv2.imshow("mask-red", masks[2])

    fg_pass_mask = cv2.bitwise_or(masks[0], masks[1])
    fg_pass_mask = cv2.bitwise_or(fg_pass_mask, masks[2])
    bg_pass_mask = cv2.bitwise_not(fg_pass_mask)

    cv2.imshow('fg_pass_mask', fg_pass_mask)
    cv2.imshow('bg_pass_mask', bg_pass_mask)

    (H, W), (h, w) = image.shape[:2], logo.shape[:2]
    x, y = (W-w)//2, (H-h)//2
    roi = image[y:y+h, x:x+w]

    cv2.imshow('roi', roi)

    foreground = cv2.bitwise_and(logo, logo, mask=fg_pass_mask)
    background = cv2.bitwise_and(roi, roi, mask=bg_pass_mask)

    dst = cv2.add(background, foreground)
    image[y:y+h, x:x+w] = dst

    sub = cv2.subtract()

    cv2.imshow('background', background); cv2.imshow('foreground', foreground)
    cv2.imshow('dst', dst); cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_reduce():
    image = cv2.imread('images/bit_test.jpg', cv2.IMREAD_GRAYSCALE)
    reduced_img = cv2.reduce(image, dim=0, rtype=cv2.REDUCE_SUM)

    cv2.imshow('reduced img', reduced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_reduce()