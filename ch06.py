from tabnanny import check
import cv2
import time
import numpy as np


# 6.2.2 - 영상 화소값 확인
def check_pixels():
    image = cv2.imread("images/pixel.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception("File error")

    (x, y), (w, h) = (180, 37), (15, 10)

    roi_img = image[y:y+h, x:x+w]

    print("roi: ")
    for row in roi_img:
        for p in row:
            print('%4d' % p, end=' ')
        print()

    cv2.rectangle(image, (x, y, w, h), 255, 1)
    cv2.imshow('image', image)
    cv2.imshow('roi', roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 6.2.3 - 행렬 가감 연산 통한 영상 밝기 변경
def modulo_vs_saturation():
    image = cv2.imread("images/bright.jpg", cv2.IMREAD_GRAYSCALE)    # 영상 읽기
    if image is None:
        raise Exception("영상 파일 읽기 오류")

    # OpenCV 함수 이용
    dst1 = cv2.add(image, 100)         # 영상 밝게 saturation 방식
    dst2 = cv2.subtract(image, 100)    # 영상 어둡게

    # numpy array 이용
    dst3 = image + 100  # 영상 밝게 modulo 방식
    dst4 = image - 100  # 영상 어둡게

    cv2.imshow("original image", image)
    cv2.imshow("dst1- bright: OpenCV", dst1)
    cv2.imshow("dst2- dark: OpenCV", dst2)
    cv2.imshow("dst3- bright: numpy", dst3)
    cv2.imshow("dst4- dark: numpy", dst4)
    cv2.waitKey(0)


# 이치화
def thresholding():
    img = cv2.imread('images/lena.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)

    # thresholding
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(img.item(y, x) < 127):
                img.itemset((y, x), 0)
            else:
                img.itemset((y, x), 255)

    cv2.imshow('bin-image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 예제 6.1.1 - 행렬 원소 접근 방법
def mat_access():
    def mat_access1(mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                k = mat[i, j]  # 원소 접근 - mat1[i][j] 방식도 가능
                mat[i, j] = k * 2  # 원소 할당

    def mat_access2(mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                k = mat.item(i, j)  # 원소 접근
                mat.itemset((i, j), k * 2)  # 원소 할당

    mat1 = np.arange(10).reshape(2, 5)
    mat2 = np.arange(10).reshape(2, 5)

    print("원소 처리 전: \n%s\n" % mat1)
    mat_access1(mat1)
    print("원소 처리 후: \n%s\n" % mat1)

    print("원소 처리 전: \n%s\n" % mat2)
    mat_access2(mat2)
    print("원소 처리 후: \n%s" % mat2)


# 예제 6.1.2 - 행렬 원소 접근
def pixel_access():
    def pixel_access1(image):
        image1 = np.zeros(image.shape[:2], image.dtype)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j]                  # 화소 접근
                image1[i, j] = 255 - pixel            # 화소 할당
        return image1

    def pixel_access2(image):
        image2 = np.zeros(image.shape[:2], image.dtype)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image.item(i, j)  # 화소 접근
                image2.itemset((i, j),  255 - pixel)  # 화소 할당
        return image2

    def pixel_access3(image):
        lut = [255 - i for i in range(256)]  # 룩업테이블 생성
        lut = np.array(lut, np.uint8)
        image3 = lut[image]
        return image3

    def pixel_access4(image):
        image4 = cv2.subtract(255, image)
        return image4

    def pixel_access5(image):
        image5 = 255 - image
        return image5

    image = cv2.imread("images/bright.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception("영상 파일 읽기 오류 발생")

    # 수행시간 체크
    def time_check(func, msg):
        start_time = time.perf_counter()
        ret_img = func(image)
        elapsed = (time.perf_counter() - start_time) * 1000
        print(msg, "수행시간 : %.2f ms" % elapsed)
        return ret_img

    image1 = time_check(pixel_access1, "[방법 1] 직접 접근 방식")
    image2 = time_check(pixel_access2, "[방법 2] item() 함수 방식")
    image3 = time_check(pixel_access3, "[방법 3] 룩업 테이블 방식")
    image4 = time_check(pixel_access4, "[방법 4] OpenCV 함수 방식")
    image5 = time_check(pixel_access5, "[방법 5] ndarray 연산 방식")

    # 결과 영상 보기
    cv2.imshow("image  - original", image)
    cv2.imshow("image1 - directly access to pixel", image1)
    cv2.imshow("image2 - item()/itemset()", image2)
    cv2.imshow("image3 - LUT", image3)
    cv2.imshow("image4 - OpenCV", image4)
    cv2.imshow("image5 - ndarray 방식", image5)
    cv2.waitKey(0)


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


def histogramStretching():
    filename = 'images/akrom.jpeg'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)

    Hist = np.zeros(256)
    ysize = img.shape[0]
    xsize = img.shape[1]
    for y in range(ysize):
        for x in range(xsize):
            Hist[img.item(y, x)] += 1

    low = 0
    high = 255
    for i in range(256):
        if Hist[i] != 0:
            low = i
            break

    for i in range(255, -1, -1):
        if Hist[i] != 0:
            high = i
            break

    for y in range(ysize):
        for x in range(xsize):
            value = round((img.item(y, x) - low)/(high - low) * 255)
            img.itemset((y, x), value)

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_equialization():
    filename = "images/akrom.jpeg"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)

    Hist = np.zeros((256))
    ysize = img.shape[0]
    xsize = img.shape[1]
    for y in range(ysize):
        for x in range(xsize):
            Hist[img.item(x, y)] += 1

    normHist = np.empty((256))
    sum = 0.0
    factor = 255.0 / (ysize * xsize)
    for i in range(256):
        sum += Hist[i]
        normHist[i] = round(sum * factor)

    for y in range(ysize):
        for x in range(xsize):
            img.itemset((y, x), normHist[img.item(y, x)])

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check_pixels()
    # modulo_vs_saturation()
    # thresholding()
    # mat_access()
    pixel_access()
    # posterize()
    # histogramStretching()
    # histogram_equialization()
