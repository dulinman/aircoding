from itertools import groupby
import cv2

def getHProjection(image):  # 水平投影
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    a = []
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
        # for y in range(h):
        if h_[y] > 40:         #35
            a.append(y)
    # print(h_)
    # print('a_h',a)
    fun = lambda x: x[1] - x[0]
    z = []
    for k, g in groupby(enumerate(a), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 1:

            num = max(l1) - min(l1)
        else:

            num = l1[0] - l1[0]
        if num < 10:
            continue
        else:
            tu = (min(l1), max(l1))
            z.append(tu)
    image = image[z[-1][0]:z[-1][1]]
    z = z[-1:]
    h_ = h_[z[0][0]:(z[0][1] + 1)]
    # cv2.namedWindow('z', cv2.WINDOW_NORMAL)
    # cv2.imshow('z', image)
    # cv2.waitKey(0)
    return h_, image, z


def getVProjection(image):  # 垂直投影

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
        if w_[x] > 3:#3
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
        if num < 5:
            continue
        else:
            tu = (min(l1), max(l1))
            z.append(tu)

    w_ = w_[z[0][0]:(z[-1][1] + 1)]
    # print(len(w_))
    return w_, z

def detetct_cha(image):
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    a = []
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
        # for y in range(h):
        if h_[y] > 1:
            a.append(y)
    line_num = len(a)
    # fun = lambda x: x[1] - x[0]
    # z = []
    # for k, g in groupby(enumerate(a), fun):
    #     l1 = [j for i, j in g]  # 连续数字的列表
    #     if len(l1) > 1:
    #         num = max(l1) - min(l1)
    #     else:
    #         num = l1[0] - l1[0]
    #     z.append(num)
    # line_num = max(z)
    return line_num




def cut_2(img):

    Position = []
    Positions = []
    H, ima, z = getHProjection(img)  # 水平投影
    (h, w) = ima.shape

    H_Start = []
    H_End = []
    Height = []
    for i in range(len(z)):
        H_Start.append(z[i][0])
        H_End.append(z[i][1])
        Height.append((z[i][1] - z[i][0]))

    # H_End.append((len(H)+z[0][0]))
    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        # 获取行图像
        cropImg = img[H_Start[i]:H_End[i], 0:w]
        W, z1 = getVProjection(cropImg)
        W_Start = 0
        W_End = 0
        st = []
        en = []
        po = []
        width = []
        # print(W[885])
        for j in range(len(z1)):
            st.append(z1[j][0])
            en.append(z1[j][1])
            width.append((z1[j][1] - z1[j][0]))

        for x in range(len(st)):
            if int(Height[i] / width[x]) < 1:
                Position.append([st[x], H_Start[i], int(en[x] - width[x] / 2), H_End[i]])
                Position.append([int(en[x] - width[x] / 2 + 1), H_Start[i], en[x], H_End[i]])
            else:
                po.append([W_Start, W_End])
                Position.append([st[x], H_Start[i], en[x], H_End[i]])
        long = len(Position)
        for y in range(long):
            line_num = detetct_cha(img[Position[y][1]: Position[y][3], Position[y][0]: Position[y][2]])
            # print(Position)
            if line_num > 0.8 * (Position[y][3]-Position[y][1]):#0.7
                Positions.append(Position[y])

    # print('**********************************')
    # for x in range(len(Positions)):
    #     cv2.rectangle(img, (Positions[x][0], Positions[x][1]), (Positions[x][2], Positions[x][3]),(255, 0, 225), 1)
    #     # 99, 36, 161, 146
    # cv2.namedWindow("cut2", cv2.WINDOW_NORMAL)
    # cv2.imshow("cut2", img)
    # cv2.waitKey(0)
    return Positions

