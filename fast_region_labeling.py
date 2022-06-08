import cv2
import numpy as np


def PeripheralHoleBoundaryTracking(memImage, cr, cc, pixel, label):
    pdir = 0  # 이전탐색방향
    ndir = 0  # 다음탐색방향
    r = cr  # row좌표
    c = cc  # column좌표
    d = [0 for i in range(8)]
    flag = False

    while True:

        '''
            clockwise rotation

            5: [r-1][c-1], 6: [r-1][c], 7: [r-1][c+1]

            4: [r][c-1],   S: [r][c], 	0: [r][c+1]

            3: [r+1][c-1], 2: [r+1][c], 1: [r+1][c+1]

            S - starting point
        '''

        d[0] = memImage[r][c + 1]
        d[1] = memImage[r + 1][c + 1]
        d[2] = memImage[r + 1][c]
        d[3] = memImage[r + 1][c - 1]
        d[4] = memImage[r][c - 1]
        d[5] = memImage[r - 1][c - 1]
        d[6] = memImage[r - 1][c]
        d[7] = memImage[r - 1][c + 1]

        if all(i == 0 for i in d):
            break

        # 탐색 방향 설정
        ndir = pdir - 3
        if (ndir == -1):
            ndir = 7
        elif (ndir == -2):
            ndir = 6
        elif (ndir == -3):
            ndir = 5

        while True:
            if (d[ndir] == pixel) or (d[ndir] == label):
                flag = False

                if pdir == 1:
                    flag = (ndir == 5)
                elif pdir == 2:
                    flag = (ndir == 5 or ndir == 6)
                elif pdir == 3:
                    flag = (ndir == 5 or ndir == 6 or ndir == 7)
                elif pdir == 4:
                    flag = (ndir == 0 or ndir == 5 or ndir == 6 or ndir == 7)
                elif pdir == 5:
                    flag = (ndir != 2 and ndir != 3 and ndir != 4)
                elif pdir == 6:
                    flag = (ndir != 3 and ndir != 4)
                elif pdir == 7:
                    flag = (ndir != 4)

                if flag:
                    memImage[r][c] = label
                pdir = ndir
                break

            else:  # 다음 탐색방향 설정
                ndir = (ndir + 1) % 8

        # 위치이동
        if ndir == 0:
            c += 1
        elif ndir == 1:
            r += 1
            c += 1
        elif ndir == 2:
            r += 1
        elif ndir == 3:
            r += 1
            c -= 1
        elif ndir == 4:
            c -= 1
        elif ndir == 5:
            r -= 1
            c -= 1
        elif ndir == 6:
            r -= 1
        elif ndir == 7:
            r -= 1
            c += 1

        if (r == cr) and (c == cc):
            break

    return memImage


def regionLabeling():
    img = cv2.imread('images/key_bw.raw.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('images/key_bw.raw.jpg', cv2.IMREAD_GRAYSCALE)
    thresh1 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1]  # 이미지 이진화
    cv2.imshow('original', thresh1)

    maxX = thresh1.shape[1]
    maxY = thresh1.shape[0]

    memImage = np.zeros((maxY, maxX))

    for y in range(0, maxY):
        for x in range(0, maxX):
            # c = 0
            # if x == 0 or y == 0 or x == (maxX-1) or y == (maxY-1):
            #     c = 0
            # else:
            c = thresh1[y][x]
            if c != 0:
                c = -255

            memImage[y][x] = c

    pixValue = 0
    label = 0

    for y in range(1, maxY - 1):
        for x in range(1, maxX - 1):
            pixValue = memImage[y][x]

            if memImage[y][x] < 0:
                # 좌측과 좌측상단이 0이거나 레이블이 없는 경우
                if (memImage[y][x - 1] <= 0) and (memImage[y - 1][x - 1] <= 0):
                    label += 1
                    memImage[y][x] = label
                    memImage = PeripheralHoleBoundaryTracking(
                        memImage, y, x, pixValue, label)
                # 좌측에 레이블이 있는 경우
                elif memImage[y][x-1] > 0:
                    memImage[y][x] = memImage[y][x-1]
                # 좌측에 레이블이 없고 좌측상단에 레이블이 있는 경우
                elif (memImage[y][x-1] <= 0) and (memImage[y-1][x-1] > 0):
                    memImage[y][x] = memImage[y-1][x-1]
                    memImage = PeripheralHoleBoundaryTracking(
                        memImage, y, x, pixValue, memImage[y-1][x-1])

    for y in range(0, maxY, 1):
        for x in range(0, maxX, 1):
            c = int(memImage[y][x] * (255 / (label+1)))  # 레이블의 수에 따라 밝기값을 균등분할
            if c == 0:
                c = 255

            for i in range(3):
                img.itemset((y, x, i), c)

    cv2.imshow('fast region labeling', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    regionLabeling()
