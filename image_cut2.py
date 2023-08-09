import cv2
import numpy as np

def Adaptive_light_correction(img):  # 基于二维伽马函数的光照不均匀图像自适应校正算法
    height = img.shape[0]
    width = img.shape[1]

    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色调(H),饱和度(S),明度(V)
    V = HSV_img[:, :, 2]
    kernel_size = 100
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



def all_image_template(image, targets, axis, rate, MIN_MATCH_COUNT):

    # MIN_MATCH_COUNT = 20  # 设置最低特征点匹配数量为10
    template = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(targets, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector创建sift检测器

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(target, None)
    kp1, des1 = sift.detectAndCompute(template, None)
    # print(des1.shape)
    # st1 = time.perf_counter()
    # print("template_image222", time.perf_counter() - st1)

    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配
    for m, n in matches:
        if m.distance < rate * n.distance:
            good.append(m)
    # for i in range(int(0.7*len(matches))):
    #     good.append(matches[i])
    # print(len(good))
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        final = 1
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        tl = (abs(int(dst[0][0][0])), abs(int(dst[0][0][1])))
        br = (abs(int(dst[2][0][0])), abs(int(dst[2][0][1])))
        target_ta = targets[abs(tl[1]):abs(br[1]), abs(tl[0]):abs(br[0]), :]
        # matchesMask = None
        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=None,
        #                    matchesMask=matchesMask,
        #                    flags=2)
        # result = cv2.drawMatches(template, kp1, targets, kp2, good, None, **draw_params)
        # cv2.namedWindow('show', cv2.WINDOW_NORMAL)
        # cv2.imshow("show", result)
        # cv2.waitKey(0)
    else:
        final = 0
        target_ta = image
        tl = (0, 0)
        br = (0, 0)

    return final, tl, br, image, target_ta




def pipei(model_img, image):
    test_img_part = []
    final_results = []
    test_img_coord = []
    axis = (0,0)
    rate = [0.8, 0.7, 0.7]
    MIN_MATCH_COUNT = [10,20,20]
    all_axis = []
    final1, tl1, br1, image1, target_parts1 = all_image_template(model_img[1], image, axis, rate[1], MIN_MATCH_COUNT[1])
    target_part1 = Adaptive_light_correction(target_parts1)
    axis_num1 = (0, 0)

    target2 = image[tl1[1]:image.shape[0], br1[0]:image.shape[1], :]
    final2, tl2, br2, image2, target_parts2 = all_image_template(model_img[2], target2, axis, rate[0], MIN_MATCH_COUNT[0])
    # final2 = 1
    tl2 = (0,0)
    # br2 = (target2.shape[0],target2.shape[1])
    # target_parts2 = target2
    print(tl2,br2)
    target_part2 = Adaptive_light_correction(target_parts2)
    z11 = br1[0]
    z21 = tl1[1]
    axis_num2 = (z11, z21)

    # target0 = image[tl1[1]:, :tl1[0], :]
    target0 = image
    final0, tl0, br0, image0, target_parts0 = all_image_template(model_img[0], target0, axis, rate[2], MIN_MATCH_COUNT[2])
    target_part0 = Adaptive_light_correction(target_parts0)
    final_results.append(final0)
    test_img_coord.append(tl0)
    test_img_coord.append(br0)
    test_img_part.append(target_part0)
    # z10 = 0
    # z20 = tl1[1]
    # axis_num0 = (z10, z20)
    axis_num0 = (0, 0)
    all_axis.append(axis_num0)
    all_axis.append(axis_num1)
    all_axis.append(axis_num2)

    final_results.append(final1)
    test_img_coord.append(tl1)
    test_img_coord.append(br1)
    test_img_part.append(target_part1)

    final_results.append(final2)
    test_img_coord.append(tl2)
    test_img_coord.append(br2)
    test_img_part.append(target_part2)

    return final_results, test_img_part, test_img_coord, all_axis


def cut_main(model_img_part, img_test):
        final_results, test_img_part, test_img_coord, all_axis = pipei(model_img_part,img_test)
        final_coord = [(test_img_coord[0][0]+all_axis[0][0], test_img_coord[0][1]+all_axis[0][1]), (test_img_coord[1][0]+all_axis[0][0], test_img_coord[1][1]+all_axis[0][1]),
                       (test_img_coord[2][0]+all_axis[1][0], test_img_coord[2][1]+all_axis[1][1]), (test_img_coord[3][0]+all_axis[1][0], test_img_coord[3][1]+all_axis[1][1]),
                       (test_img_coord[4][0]+all_axis[2][0], test_img_coord[4][1]+all_axis[2][1]), (test_img_coord[5][0]+all_axis[2][0], test_img_coord[5][1]+all_axis[2][1])]
        # cv2.namedWindow('z', cv2.WINDOW_NORMAL)
        # cv2.imshow("z" , test_img_part[0])
        # cv2.waitKey()
        return final_results, model_img_part, test_img_part, final_coord


