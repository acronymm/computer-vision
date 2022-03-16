import cv2
import numpy as np

BASE_DIR = './images/'

def showImage(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)

    # thresholding
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         if(img.item(y,x) < 127):
    #             img.itemset((y, x), 0)
    #         else:
    #             img.itemset((y,x), 255)

    # cv2.imshow('bin-image', img)

    # channel 분리 
    b, g, r = cv2.split(img)
    cv2.imshow('b image', b)
    cv2.imshow('g image', g)
    cv2.imshow('r image', r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showVideo():
    try:
        cap = cv2.VideoCapture(0)
    except:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # showImage(BASE_DIR + 'lena.jpg')
    showVideo()