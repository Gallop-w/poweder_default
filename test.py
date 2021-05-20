# -*- encoding: utf-8 -*-
# can i push you ?
# can i pull you ?
import os
import cv2
import numpy as np

def get_file_path(root_path):
    filenames = os.listdir(root_path)
    for i in filenames:
        if '_0' in i:
            filenames.remove(i)

    return filenames

def cv_show(img):
    cv2.namedWindow('test', 0)
    # cv2.resizeWindow('test', 1000, 500)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
root_path = r'./5.15/1-1647/images'
filelist = get_file_path(root_path)

ret = []
cnt = 0
for img_id in filelist:
    img = cv2.imread(os.path.join('./5.15/1-1647/images/', img_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[350:1100, 700:1200]
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    closed = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    draw_contours = cv2.drawContours(closed.copy(), contours, -1, (0, 255,255), 1)
    cv_show(draw_contours)
    area_max = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > area_max:
            area_max = area
    ret.append('%s: %d'%(img_id, area_max))
    cnt += 1
    if cnt == 1000:
        break

txtfile = r'ret.txt'
file = open(txtfile, 'w')
file.writelines([line + '\n' for line in ret])
file.close()
