import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def cv_show(img):
    cv2.namedWindow('test', 0)
    # cv2.resizeWindow('test', 1000, 500)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gray_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# 获取文件列表,滤除铺粉图像
def get_file_path(root_path):
    filenames = os.listdir(root_path)
    for i in filenames:
        if '_0' in i:
            filenames.remove(i)

    return filenames

def callback():
    pass
    # 滑动块创建
def trackbar(img):


    cv2.namedWindow('image')

    cv2.createTrackbar('C', 'image', 1, 15, callback)

    while(True):
        c = cv2.getTrackbarPos('C', 'image')

        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, c)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cv2.imshow('image', closed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 图片读取
root_path = r'./5.15/1-1647/images'
filelist = get_file_path(root_path)
# print(filelist)



for img in filelist:
    image = cv2.imread(os.path.join('./5.15/1-1647/images/', img))
    cropped = image[350:1100, 700:1200]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    # vx = cv2.Sobel(gray, -1, 1, 0)
    # vy = cv2.Sobel(gray, -1, 0, 1)
    # v1 = cv2.addWeighted(vx, 0.5, vy, 0.5, 0)
    # cv_show(v1)
    # # v1 = cv2.Canny(gray, 0, 20)
    # contours, hierarchy = cv2.findContours(v1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # draw_contours = cv2.drawContours(gray.copy(), contours, -1, (0, 0, 255), 1)
    # cv_show(draw_contours)





    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    # cv_show(thresh)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    trackbar(gray)
    pass

# 图片裁剪
# cropped = image[350:1100 , 700:1200] #y0:y1, x0:x1
# a = cropped.shape
# cv_show(cropped)   #   ,thresh1

# 灰度转换
# gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# 滤波处理
# gaussian = cv2.GaussianBlur(gray, (1, 1), 0)

# 阈值处理
# ret, thresh1 = cv2.threshold(gaussian, 0, 255, cv2.THRESH_OTSU)
# thresh1 = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5) # 自适应滤波处理yyds
# ret, thresh2 = cv2.threshold(cropped, 100, 255, cv2.THRESH_BINARY) 效果不好，有误判区域
# res = np.hstack((cropped, thresh1))
# cv_show(res)

#canny算子
v1 = cv2.Canny(thresh1, 100, 255)

# 边缘检测
contours, hierarchy = cv2.findContours(v1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

draw_contours = cv2.drawContours(cropped.copy(), contours, -1, (0, 0,255), 1)
# contour_thresh_copy = v1.copy()


# 结果显示
res = np.hstack((cropped, draw_contours ))
cv_show(res)

# if __name__ == '__main__':

# plt.hist(gaussian.ravel(), 256) # 一维化图像， 256=bins，间隔数量
# plt.show()

