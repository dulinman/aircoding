import time
from itertools import groupby
from PaddleOcrUtil import PaddleOcr
import cv2
import numpy as np

POCR = PaddleOcr()
def Adaptive_light_correction(img):  # 基于二维伽马函数的光照不均匀图像自适应校正算法
    height = img.shape[0]
    width = img.shape[1]

    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色调(H),饱和度(S),明度(V)
    V = HSV_img[:, :, 2]
    kernel_size = min(height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1  # 必须是正数和奇数
    SIGMA1 = 150
    SIGMA2 = 800
    SIGMA3 = 2500
    q = np.sqrt(2.0)
    F = np.zeros((height, width, 3), dtype=np.float64)
    F[:, :, 0] = cv2.GaussianBlur(V, (kernel_size, kernel_size), SIGMA1 / q)
    F[:, :, 1] = cv2.GaussianBlur(V, (kernel_size, kernel_size), SIGMA2 / q)
    F[:, :, 2] = cv2.GaussianBlur(V, (kernel_size, kernel_size), SIGMA3 / q)
    F_mean = np.mean(F, axis=2)
    average = np.mean(F_mean)
    gamma = np.power(0.3, np.divide(np.subtract(average, F_mean), average))
    out = np.power(V / 255.0, gamma) * 255.0
    HSV_img[:, :, 2] = out
    light_correct_img = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR)
    return light_correct_img


def arr_sort(arr):
    index_0 = []
    for i in range(len(arr)):
        index_0.append(arr[i][0])
    index = np.array(index_0)
    a = np.average(index, axis = 1)
    arrSortedIndex = np.argsort(a)
    # print(arrSortedIndex)
    re = arr[arrSortedIndex]
    return re



def getVProjection(image,thre,i):  # 垂直投影

    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    a = []
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
        # for y in range(len(w_)):
        if w_[x] > thre[i][0]:
            a.append(x)
    # print('a_v',a)
    fun = lambda x: x[1] - x[0]
    z = []
    for k, g in groupby(enumerate(a), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 1:
            # scop = str(min(l1)) + '-' + str(max(l1))  # 将连续数字范围用"-"连接
            num = max(l1) - min(l1)

        else:
            # scop = l1[0]
            num = l1[0] - l1[0]
        if num < thre[i][1]:
            continue
        else:
            tu = (min(l1), max(l1))
            z.append(tu)

    w_ = w_[z[0][0]:(z[-1][1] + 1)]
    # print(len(w_))
    return w_, z


# 改为直接传递二值化后的图像，且进行开闭运算后
def cut_0(img):

    # img = Adaptive_light_correction(img)
    result = POCR.ppocr_seg(img)
    result = arr_sort(np.array(result))

    Position = []

    thre = [(9, 7), (9, 7)]
    # for i in range(len(H_Start)):
    for i in range(len(result)):
        line_axis = result[i]
        array_axis = np.array(line_axis)
        min_valus = np.min(array_axis, axis=0)
        max_valus = np.max(array_axis, axis=0)
        x_min = int(min_valus[0])
        x_max = int(max_valus[0])
        y_min = int(min_valus[1])
        y_max = int(max_valus[1])
        line_image = img[y_min:y_max,x_min:x_max]
        ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  ##腐蚀预处理，确定处理核的大小,矩阵操作
        line_image = cv2.dilate(line_image, ker, iterations=1)  # 进行腐蚀操作

        W, z1 = getVProjection(line_image,thre,i)
        # Wend = 0
        W_Start = 0
        W_End = 0
        st = []
        en = []
        po = []
        width = []
        # a = 0
        for j in range(len(z1)):
            st.append(z1[j][0])
            en.append(z1[j][1])
            width.append((z1[j][1] - z1[j][0]))
        for j in range(len(st)):
            if i == 0:
                # if line_image.shape[0] / width[j] < 6:
                Position.append([(st[j]+x_min), y_min, (en[j]+x_min), y_max])

            elif i == 1:
                # print(line_image.shape[0] / width[j])
                if line_image.shape[0] / width[j] < 1.4:
                    # print(int(Height[i] / width[x]))
                    Position.append([(st[j]+x_min), y_min, (int(en[j]-width[j]/2)+x_min), y_max])
                    Position.append([(int(en[j]-width[j]/2+1)+x_min), y_min, (en[j]+x_min), y_max])
                # elif Height[i] / width[j] < 5.6:
                # elif st[j] > st_line[0] and en[j] < en_line[-1] and line_image.shape[0] / width[j] < 6:
                else :
                    po.append([W_Start, W_End])
                    Position.append([(st[j]+x_min), y_min, (en[j]+x_min), y_max])

    # print('******************************')
    # for x in range(len(Position)):
    #     cv2.rectangle(img, (Position[x][0], Position[x][1]), (Position[x][2], Position[x][3]), (255, 0, 225), 1)
    #     # 99, 36, 161, 146
    # # print(len(Position))
    # cv2.namedWindow("cut", cv2.WINDOW_NORMAL)
    # cv2.imshow("cut", img)
    # cv2.waitKey(0)

    return Position

