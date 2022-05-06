from re import sub
from tabnanny import check
from this import d
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt


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


# LUT 적용의 예
def custom_lut():
    img = cv2.imread('images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)

    # lookup table
    lut = np.arange(255, -1, -1, dtype="uint8")

    ysize = img.shape[0]
    xsize = img.shape[1]
    for y in range(ysize):
        for x in range(xsize):
            img.itemset((y, x), lut[img.item(y, x)])

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2.LUT()
def openCV_lut():
    img = cv2.imread('images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)

    lut = np.arange(255, -1, -1, dtype='uint8')

    result = cv2.LUT(img, lut)
    cv2.imshow('result', result)
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
    for idx, thres in enumerate(range(60, 1, -10), start=1):
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


# cv2.subtract()
def subtract():
    file1 = "images/ic_ref.raw.jpg"
    file2 = "images/ic_test.raw.jpg"

    img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image1', img1)
    img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image2', img2)

    result = cv2.subtract(img1, img2)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_synthesis():
    image1 = cv2.imread("images/add1.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("images/add2.jpg", cv2.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        raise Exception("영상 파일 읽기 오류 발생")

    # 영상 합성
    alpha, beta = 0.6, 0.7
    add_img1 = cv2.add(image1, image2)
    add_img2 = cv2.add(image1 * alpha, image2 * beta)
    add_img2 = np.clip(add_img2, 0, 255).astype("uint8")
    add_img3 = cv2.addWeighted(image1, alpha, image2, beta, 0)

    titles = ['image1', 'image2', 'add_img1', 'add_img2', 'add_img3']
    for t in titles:
        cv2.imshow(t, eval(t))
    cv2.waitKey(0)


def contrast():
    image = cv2.imread("images/contrast.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception("영상 파일 읽기 오류 발생")

    noimage = np.zeros(image.shape[:2], image.dtype)
    avg = cv2.mean(image)[0]/2.0

    dst1 = cv2.scaleAdd(image, 0.2, noimage) + 20
    dst2 = cv2.scaleAdd(image, 2.0, noimage)
    dst3 = cv2.addWeighted(image, 0.5, noimage, 0, avg)
    dst4 = cv2.addWeighted(image, 2.0, noimage, 0, -avg)

    cv2.imshow("image", image)
    cv2.imshow("dst1 - decrease contrast", dst1)
    cv2.imshow("dst2 - increase contrast", dst2)
    cv2.imshow("dst3 - decrease contrast using average", dst3)
    cv2.imshow("dst4 - increase contrast using average", dst4)

    cv2.imwrite("dst.jpg", dst1)
    cv2.waitKey(0)


def otsu_algorithm():
    filename = "images/Coin.bmp"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    # Create Histogram
    Hist = np.zeros((256))
    ysize = img.shape[0]
    xsize = img.shape[1]
    for y in range(ysize):
        for x in range(xsize):
            Hist[img.item(y, x)] = Hist[img.item(y, x)] + 1

    Prob = np.empty((256), dtype="float32")
    for i in range(256):
        Prob[i] = Hist[i] / img.size

    wsv_min = float('inf')

    for t in range(256):
        # 두 집단의 확률 q1, q2 계산
        q1 = q2 = 0.0
        for i in range(t):
            q1 = q1 + Prob[i]
        for i in range(t, 256):
            q2 = q2 + Prob[i]
        if q1 == 0 or q2 == 0:
            continue

        # 두 집단의 평균 u1, u2 계산
        u1 = u2 = 0.0
        for i in range(t):
            u1 = u1 + i * Prob[i]
        u1 = u1 / q1
        for i in range(t, 256):
            u2 = u2 + i * Prob[i]
        u2 = u2 / q2

        # 두 집단의 분산 s1, s2 계산
        s1 = s2 = 0.0
        for i in range(t):
            s1 = s1 + pow(i-u1, 2) * Prob[i]
        s1 = s1 / q1
        for i in range(t, 256):
            s2 = s2 + pow(i-u2, 2) * Prob[i]
        s2 = s2 / q2

        # 수식 wsv를 계산하고 임계치 wsv_t를 결정
        wsv = q1 * s1 + q2 * s2
        if wsv < wsv_min:
            wsv_min = wsv
            wsv_t = t
        # end-for t

    # Thresholding by wsv_t
    for y in range(ysize):
        for x in range(xsize):
            if img.item(y, x) < wsv_t:
                img.itemset((y, x), 0)
            else:
                img.itemset((y, x), 255)
            # end-for x
        # end-for y

    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2.threshold()
def openCV_threshold():
    filename = "images/Coin.bmp"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret, result1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    ret, result2 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    ret, result3 = cv2.threshold(img, 128, 255, cv2.THRESH_TRUNC)
    ret, result4 = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO)
    ret, result5 = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO_INV)
    ret, result6 = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('BINARY', result1)
    cv2.imshow('BINARY_INV', result2)
    cv2.imshow('TRUNC', result3)
    cv2.imshow('TOZERO', result4)
    cv2.imshow('TOZERO_INV', result5)
    cv2.imshow('OTSU', result6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# cv2.adaptiveThreshold()
def openCV_adaptiveThreshold():
    filename = "images/VIN0.raw.jpg"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret, result = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result1 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    result2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    cv2.imshow('image', img)
    cv2.imshow('OTSU', result)
    cv2.imshow('MEAN_C', result1)
    cv2.imshow('GAUSSIAN_C', result2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_hist():
    filename = "images/lena.jpg"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)

    Hist = np.zeros((256))
    maxValue = 0
    ysize = img.shape[0]
    xsize = img.shape[1]
    for y in range(ysize):
        for x in range(xsize):
            Hist[img.item(y, x)] = Hist[img.item(y, x)] + 1
            if(Hist[img.item(y, x)] > maxValue):
                maxValue = Hist[img.item(y, x)]

    # Draw the histogram on imgHist
    imgHist = np.zeros((256, 256), dtype="uint8")
    for i in range(256):
        value = Hist[i]
        normalized = math.floor(value * 255 / maxValue)
        for j in range(255, 255-normalized, -1):
            imgHist.itemset((j, i), 255)
    cv2.imshow('histogram', imgHist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def openCV_calcHist():
    filename = "images/lena.jpg"
    gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.plot(hist, color='black')
    color_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    for i, c in enumerate(('blue', 'green', 'red')):
        # 히스토그램 계산
        hist = cv2.calcHist([color_img], [i], None, [256], [0, 256])
        # matplotlib로 히스토그램 그리기
        plt.plot(hist, color=c)
        plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 예제 6.3.1 - 영상 히스토그램 계산
def calc_hist_opencv():
    def calc_histo(image, hsize, ranges=[0, 256]):  # 행렬 원소의 1차원 히스토그램 계산
        hist = np.zeros((hsize, 1), np.float32)  # 히스토그램 누적 행렬
        gap = ranges[1] / hsize  # 계급 간격

        for i in range(image.shape[0]):  # 2차원 행렬 순회 방식
            for j in range(image.shape[1]):
                idx = int(image.item(i, j) / gap)
                hist[idx] += 1
        return hist

    image = cv2.imread("images/pixel.jpg", cv2.IMREAD_GRAYSCALE)  # 영상 읽기
    if image is None:
        raise Exception("영상 파일 읽기 오류 발생")

    hsize, ranges = [32], [0, 256]                  # 히스토그램 간격수, 값 범위
    gap = ranges[1]/hsize[0]
    ranges_gap = np.arange(0, ranges[1]+1, gap)
    hist1 = calc_histo(image, hsize[0], ranges)  # User 함수
    hist2 = cv2.calcHist([image], [0], None, hsize, ranges)  # OpenCV 함수
    hist3, bins = np.histogram(image, ranges_gap)

    print("User 함수: \n", hist1.flatten())                # 행렬을 벡터로 변환하여 출력
    print("OpenCV 함수: \n", hist2.flatten())                # 행렬을 벡터로 변환하여 출력
    print("numpy 함수: \n", hist3)                           # 행렬을 벡터로 변환하여 출력

    cv2.imshow("image", image)
    cv2.waitKey(0)


def histogramStretching():
    filename = 'images/Coin.bmp'
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
    filename = "images/lena.jpg"
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


def draw_histo(hist, shape=(200, 256)):
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
    gap = hist_img.shape[1]/hist.shape[0]

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        roi = (x, 0, w, int(h))
        cv2.rectangle(hist_img, roi, 0, cv2.FILLED)
    return cv2.flip(hist_img, 0)


def histogramStretching2():
    def search_value_idx(hist, bias=0):
        for i in range(hist.shape[0]):
            idx = np.abs(bias - i)
            if hist[idx] > 0:
                return idx
        return -1

    image = cv2.imread("images/hist_stretch.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception("영상 파일 읽기 오류")

    bsize, ranges = [64], [0, 256]
    hist = cv2.calcHist([image], [0], None, bsize, ranges)

    bin_width = ranges[1]/bsize[0]
    high = search_value_idx(hist, bsize[0] - 1) * bin_width
    low = search_value_idx(hist, 0) * bin_width

    idx = np.arange(0, 256)
    idx = (idx - low) * 255/(high-low)
    idx[0:int(low)] = 0
    idx[int(high+1):] = 255

    dst = cv2.LUT(image, idx.astype('uint8'))

    hist_dst = cv2.calcHist([dst], [0], None, bsize, ranges)
    hist_img = draw_histo(hist, (200, 360))
    hist_dst_img = draw_histo(hist_dst, (200, 360))

    print("high_value = ", high)
    print("low_value = ", low)
    cv2.imshow("image", image)
    cv2.imshow("hist_img", hist_img)
    cv2.imshow("dst", dst)
    cv2.imshow("hist_dst_img", hist_dst_img)
    cv2.waitKey(0)


# 예제 6.3.6 - 히스토그램 평활화
def histogram_equalizer():
    image = cv2.imread("images/equalize.jpg", cv2.IMREAD_GRAYSCALE)  # 영상 읽기
    if image is None:
        raise Exception("영상 파일 읽기 오류")

    bins, ranges = [256], [0, 256]
    hist = cv2.calcHist([image], [0], None, bins, ranges)    # 히스토그램 계산

    # 히스토그램 누적합 계산
    accum_hist = np.zeros(hist.shape[:2], np.float32)
    accum_hist[0] = hist[0]
    for i in range(1, hist.shape[0]):
        accum_hist[i] = accum_hist[i - 1] + hist[i]

    accum_hist = (accum_hist / sum(hist)) * 255                 # 누적합의 정규화
    dst1 = [[accum_hist[val] for val in row] for row in image]  # 화소값 할당
    dst1 = np.array(dst1, np.uint8)

    # #numpy 함수 및 룩업 테이블 사용
    # accum_hist = np.cumsum(hist)                      # 누적합 계산
    # cv2.normalize(accum_hist, accum_hist, 0, 255, cv2.NORM_MINMAX)  # 정규화
    # dst1 = cv2.LUT(image, accum_hist.astype("uint8"))  #룩업 테이블로 화소값할당

    dst2 = cv2.equalizeHist(image)                # OpenCV 히스토그램 평활화
    hist1 = cv2.calcHist([dst1], [0], None, bins, ranges)   # 히스토그램 계산
    hist2 = cv2.calcHist([dst2], [0], None, bins, ranges)   # 히스토그램 계산
    hist_img = draw_histo(hist)
    hist_img1 = draw_histo(hist1)
    hist_img2 = draw_histo(hist2)

    cv2.imshow("image", image)
    cv2.imshow("hist_img", hist_img)
    cv2.imshow("dst1_User", dst1)
    cv2.imshow("User_hist", hist_img1)
    cv2.imshow("dst2_OpenCV", dst2)
    cv2.imshow("OpenCV_hist", hist_img2)
    cv2.waitKey(0)


if __name__ == '__main__':
    # Part 1
    # check_pixels()
    # modulo_vs_saturation()
    # thresholding()
    # mat_access()
    # pixel_access()
    # custom_lut()
    # openCV_lut()
    # posterize()
    # subtract()
    # image_synthesis()
    # contrast()
    # otsu_algorithm()
    # openCV_threshold()
    # openCV_adaptiveThreshold()

    # Part 2
    # draw_hist()
    # openCV_calcHist()
    # calc_hist_opencv()
    # histogramStretching2()
    # histogram_equialization()
    histogram_equalizer()
