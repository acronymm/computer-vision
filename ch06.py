from tabnanny import check
import cv2
import numpy as np


def check_pixels():
    image = cv2.imread("images/pixel.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None: raise Exception("File error")

    (x, y), (w, h) = (180, 37), (15, 10)

    roi_img = image[y:y+h, x:x+w]

    print("roi: ")
    for row in roi_img:
        for p in row:
            print('%4d' % p, end=' ')
        print()

    cv2.rectangle(image, (x,y,w,h), 255, 1)
    cv2.imshow('image', image)
    cv2.imshow('roi', roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createPosterizingLUT(thres):
    lut = np.arange(256)
    for i in range(256):
        lut[i] = lut[i] // thres * thres
    return lut


def posterize():
    image = cv2.imread('images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)

    # result = cv2.LUT(image, createLUT())

    ysize = image.shape[0]
    xsize = image.shape[1]
    for idx, thres in enumerate(range(60, 1, -10)):
        result_img = np.array(image)
        lut = createPosterizingLUT(thres)

        for y in range(ysize):
            for x in range(xsize):
                result_img.itemset((y, x), lut[result_img.item(y, x)])

        window_name = f'result {thres}'
        cv2.namedWindow(window_name)
        moveX = (idx * xsize) % 1920
        moveY = 0 if 1920 - (idx * xsize) >= xsize else ysize
        cv2.moveWindow(window_name, moveX, moveY)

        cv2.imshow(f'result {thres}', result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check_pixels()
    posterize()

