import cv2
import numpy as np
import os
import time

"""
本段程序实现图像亮度评分,具体使用方法如下
img = cv2.imread('imgpath', 1)
print(brightness_decetion(img)
"""


def brightness_decetion(img):
    # 返回亮度评分,数值在0~100;数值越高表示越量
    brightness_score = np.multiply(np.divide(np.sum(img), img.size), 0.39)
    brightness_score = int(brightness_score)
    if brightness_score < 20:
        brightness_degree = 'very dark'
    elif brightness_score > 80:
        brightness_degree = 'very bright'
    else:
        brightness_degree = 'normal'
    brightness_score = str(int(brightness_score))
    return brightness_score, brightness_degree


def main():
    # test code #
    rootdir = 'D:\\graduation_design_imgfile\\bright_test'
    file_list = os.listdir(rootdir)
    for i in file_list:
        time_start = time.perf_counter()
        file_name = '{0}\\{1}'.format(rootdir, i)
        img = cv2.imread(file_name, 1)
        print('{0}|{1}'.format(i, brightness_decetion(img)))
        elasped_time = time.perf_counter() - time_start
        print('elaspe {0} secend'.format(elasped_time))
        pass
    # test code #
    # example code #
    # img = cv2.imread('D:\\graduation_design_imgfile\\555.png')
    # print(brightness_decetion(img))
    # example code #


if __name__ == "__main__":
    main()
