import cv2
import numpy as np
import pywt
import pywt.data
import time
"""
本段算法完成高斯噪声方差估计,可以认为当sigma>10时存在噪声异常
"""

def gaussnoise(img):
    # 作用,返回图像高斯噪声方差估计,效果不错
    # 参考文献: Ideal spatial adaptation by wavelet shrinkage
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    sigma = np.divide(np.median(np.abs(HH)), 0.6745)
    if sigma < 5:
        degree = 'normal'
    elif sigma > 10:
        degree = 'very bad'
    else:
        degree = 'bad'
    sigma = str(int(sigma))
    return sigma, degree
    pass


def main():

    for i in range(1, 10):
        for j in range(1, 6):
            time_start = time.perf_counter()
            filename = "D:\\graduation_design_imgfile\\TID2013\\distorted_images\\I0{0}_01_{1}.bmp".format(i, j)
            img1 = cv2.imread(filename, 1)
            sigma1 = gaussnoise(img1)
            print('img{0}{1} score| {2}'.format(i, j, sigma1))
            elasped_time = time.perf_counter() - time_start
            print('elaspe {0} secend'.format(elasped_time))

if __name__ == "__main__":
    main()
