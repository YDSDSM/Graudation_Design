import cv2
import math
import numpy as np
import os
import time
# 本段程序用两种方法实现了偏色检测,方法一具有普适性但是准确率一般 方法2效果好一点但是依赖门限值.
# 偏色本身就是一个很主观的东西,不深究了.
# cast值不大于1.5表示正常图片  这个方法其实不大像 如果图像整体基调就是偏红的话 输出值就比较大
# 利用wiki上的图片检测 测不出来
# 本法有一个计算细节,即等效圆的中心在a=0, b=0的位置, 这就意味着a,b通道的图片要数值取值范围要从[0,255]移到[-128,127]的位置.
# 方法2与方法1有些类似，计算的东西都差不多，方法2实质上是对方法1的优化，但是优化的方式很糟糕，不具有普适性。
# 不过方法2针对特定的场景检测效果是不错的，受图片大小影响比较大
# 经多次实践检验,方法2简直是垃圾!


def colorcastdetection1(img):
    #作用,判断图片偏色程度,并且根据mean_a 与mean_b的比值 可以判断具体的偏色类型
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img)
    h, w, _ = img.shape
    mean_a = a_channel.sum()/(h*w)-128
    mean_b = b_channel.sum()/(h*w)-128
    histA = [0]*256
    histB = [0]*256
    for i in range(h):
        for j in range(w):
            ta = a_channel[i][j]
            tb = b_channel[i][j]
            histA[ta] += 1
            histB[tb] += 1
    msqA = 0
    msqB = 0
    for y in range(256):
        msqA += float(abs(y-128-mean_a))*histA[y]/(w*h)
        msqB += float(abs(y - 128 - mean_b)) * histB[y] / (w * h)
    K = math.sqrt(mean_a*mean_a+mean_b*mean_b)/(math.sqrt(msqA*msqA+msqB*msqB)+0.1)
    castcolor = 'normal'
    if K > 1:
        if np.logical_and(mean_a > mean_b, mean_a > -mean_b):
            castcolor = 'red'
        elif np.logical_and(mean_a > mean_b, mean_a < -mean_b):
            castcolor = 'blue'
        elif np.logical_and(mean_a < mean_b, mean_a > -mean_b):
            castcolor = 'yellow'
        else:
            castcolor = 'green'
    K = str(int(K))
    return K, castcolor


def colorcastdetection2(img1):
    # 0分无偏色 1-2分引起注意 3分以上图应该是不行了。
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)  # 转到CIELAB空间,这个空间处理颜色问题比较好,符合人眼特性.   function(1~4)
    l_channel, a_channel, b_channel = cv2.split(img1)
    a_channel = np.subtract(a_channel, 128)
    b_channel = np.subtract(b_channel, 128)
    mean_a = np.mean(a_channel)
    mean_b = np.mean(b_channel)
    D = np.sqrt(np.add(np.power(mean_a, 2), np.power(mean_b, 2)))  # function(5)
    M_a = np.mean(np.abs(np.subtract(a_channel, mean_a)))
    M_b = np.mean(np.abs(np.subtract(b_channel, mean_b)))
    M = np.sqrt(np.add(np.power(M_a, 2), np.power(M_b, 2)))
    K = np.divide(D, M)  # 与法1的K一致
    sigma2 = np.var(img1_gray)
    min1 = np.array([np.subtract(sigma2, 2636), 1])  # 这里的常数是问题的关键，选择适当，就可以不误伤队友， 据此可以测试误判率。
    min1 = np.min(min1)
    max1 = np.array([np.abs(min1), 1])
    max1 = np.max(max1)
    K0 = np.divide(D, np.multiply(M, max1))
    # if np.logical_and(mean_a > mean_b, mean_a > -mean_b):
    #     castcolor = 'red'
    # elif np.logical_and(mean_a > mean_b, mean_a < -mean_b):
    #     castcolor = 'blue'
    # elif np.logical_and(mean_a < mean_b, mean_a > -mean_b):
    #     castcolor = 'yellow'
    # else:
    #     castcolor = 'green'
    return K0, K


def main():
    # example code #
    # img1 = cv2.imread('D:\\graduation_design_imgfile\\555.png', 1)
    # result1, str1 = colorcastdetection1(img1)
    # print(result1)
    # print(str1)
    # example code
    # test code #
    rootdir = 'D:\\graduation_design_imgfile\\brightness_threshold'
    file_list = os.listdir(rootdir)
    for i in file_list:
        time_start = time.perf_counter()
        file_name = '{0}\\{1}'.format(rootdir, i)
        img = cv2.imread(file_name, 1)
        score, degree = colorcastdetection1(img)
        print('{0}|{1}|{2}'.format(i, score, degree))
        elasped_time = time.perf_counter() - time_start
        print('elaspe {0} secend'.format(elasped_time))
        pass
    # test code #

if __name__ == "__main__":
    main()
