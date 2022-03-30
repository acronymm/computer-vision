import numpy as np
import cv2

def windowControlEx():
    image = np.ones((200, 400), np.uint8)
    # image[:] = 200

    title1, title2 = 'Position1', 'Position2'
    cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title1, 150, 150)
    cv2.moveWindow(title2, 400, 50)

    cv2.imshow(title1, image)
    cv2.imshow(title2, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawLineRect():
    blue, green, red = (255,0,0), (0,255,0), (0,0,255)
    image = np.zeros((400, 600, 3), np.uint8)
    image[:] = (255, 255, 255)

    pt1, pt2 = (50, 50), (250, 150)
    pt3, pt4 = (400, 150), (500, 50)
    roi = (50, 200, 200, 100)

    # line
    cv2.line(image, pt1, pt2, red)
    cv2.line(image, pt3, pt4, green, 3, cv2.LINE_AA)

    # rectangle
    cv2.rectangle(image, pt1, pt2, blue, 3, cv2.LINE_4)
    cv2.rectangle(image, roi, red, 3, cv2.LINE_8)
    cv2.rectangle(image, (400, 200, 100, 100), green, cv2.FILLED)

    cv2.namedWindow('Line & Rectangle')
    cv2.imshow('Line & Rectange', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_image():
    image = cv2.imread('images/read_color.jpg', cv2.IMREAD_COLOR)
    if image is None: raise Exception('Image Reading Error')

    params_jpg = (cv2.IMWRITE_JPEG_QUALITY, 10)
    params_png = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    cv2.imwrite('images/write_test1.jpg', image)
    cv2.imwrite('images/write_test2.jpg', image, params_jpg)
    cv2.imwrite('images/write_test3.png', image, params_png)
    cv2.imwrite('images/write_test4.bmp', image)
    print('end of write_image()')


if __name__ == '__main__':
    # windowControlEx()
    # drawLineRect()
    write_image()