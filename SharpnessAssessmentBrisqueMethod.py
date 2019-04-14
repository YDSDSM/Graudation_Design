import os
import cv2
import numpy as np
from scipy import signal
from scipy.special import gamma
from PIL import Image

"""
本段程序基于BRISQUE方法实现图像清晰度评分,具体使用方法如下
img = cv2.imread('imgpath', 1)
features = get_feature(img)
print(compute_score(features))
"""


def gauss2D(size, sigma):
    """
    :param size: 模板大小
    :param sigma: 方差
    :return: 高斯模板
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def MSCN_img(img):
    """
    :param img: 输入特性
    :return: 返回mean subtracted contrast normalized(MSCN)图像
    功能: 计算MSCN图像,此图像在特征域能区分清晰图像与失真图像,将主要用于后续图像特征计算.
    已正确编写
    """
    img = img.astype('float64')
    window = gauss2D(7, 7/6)  # 生成高斯模板
    mu = signal.convolve2d(img, window, mode="same")  # 正确 function(2)
    mu_sq = np.multiply(mu, mu)  # 正确 用于计算function(3)
    img_sq = np.multiply(img, img)  #  4/11 debug到这里,正在查MSCN的输出是否正确
    img_sq_filtered = signal.convolve2d(img_sq, window, mode="same")
    sigma = np.sqrt(np.abs(np.subtract(img_sq_filtered, mu_sq)))  # 有问题
    mscn_img = np.divide(np.subtract(img, mu), np.add(sigma, 1))
    return mscn_img
    pass


def estimate_ggdparam(mscn_img):
    """
    :param mscn_img: MSCN图像
    :return: GGD估计参数,即(α,σ)
    作用:计算GGD参数,即"function(4)"涉及的两个参数,两个参数作为feature的一部分,存储后用于后续quanlity score的计算.
    """
    gam = np.arange(0.2, 10, 0.001)
    r_gam = np.divide(np.multiply(gamma(np.divide(1, gam)), gamma(np.divide(3, gam))),
                      np.power(gamma(np.divide(2, gam)), 2))
    sigma_sq = np.mean(np.power(mscn_img, 2))  #
    sigma = np.sqrt(sigma_sq)  # 写入到特征值中的是sigma^2 注意!!
    E = np.mean(np.abs(mscn_img))
    rho = np.divide(sigma_sq, np.power(E, 2))
    array1 = np.abs(np.subtract(rho, r_gam))
    array_position = np.where(array1 == np.min(array1))
    gamparam = gam[array_position[0]]
    return gamparam, sigma
    pass


def estimate_aggdparam(mscn_shifted):
    """
    :param mscn_shifted: 移位相乘图,即function(7-10）
    :return: AGGD估计参数,即(v, σsubl, σsubr)
    作用:计算AGGD参数,即function(12)涉及的三个参数,还有第四个参数<function(15)>,可以直接计算,就不囊括在这里面了.
    """
    gam = np.arange(0.2, 10, 0.001)
    r_gam = np.divide(np.power(gamma(np.divide(2, gam)), 2),
                      np.multiply(gamma(np.divide(1, gam)), gamma(np.divide(3, gam))))
    leftstd = np.sqrt(np.mean(np.power(mscn_shifted[mscn_shifted < 0], 2)))
    rightstd = np.sqrt(np.mean(np.power(mscn_shifted[mscn_shifted > 0], 2)))
    gammahat = np.divide(leftstd, rightstd)
    rhat = np.divide(np.power(np.mean(np.abs(mscn_shifted)), 2),
                     np.mean(np.power(mscn_shifted, 2)))  # 有误
    rhatnorm = np.divide(np.multiply(rhat, np.multiply(np.add(np.power(gammahat, 3), 1), np.add(gammahat, 1))),
                         np.power(np.add(np.power(gammahat, 2), 1), 2))  # 有误
    array1 = np.power(np.subtract(r_gam, rhatnorm), 2)
    array_position = np.where(array1 == np.min(array1))
    alpha = gam[array_position[0]]
    return alpha, leftstd, rightstd  # 后俩已经算对了 4/12 15:02
    pass


def pairs(mscn_img1):
    """
    :param mscn_img1: MSCN图片,用于生成H, V, D1, D2图片
    :return: H, V, D1, D2图片
    功能:生成H, V, D1, D2图片,用于估计AGGD参数
    """
    VM = np.roll(mscn_img1, 1, 0)  # function(8)的中间变量
    D1M = np.roll(VM, 1, 1)  # function(9)的中间变量
    HM = np.roll(D1M, -1, 0)  # function(7)的中间变量
    D2M = np.roll(HM, -1, 0)  # function(10)的中间变量 以上四句不可调换.
    H = np.multiply(mscn_img1, HM)
    V = np.multiply(mscn_img1, VM)
    D1 = np.multiply(mscn_img1, D1M)
    D2 = np.multiply(mscn_img1, D2M)
    return H, V, D1, D2


def get_feature(img1):
    """
    :param img1: 输入图片, 只接受彩图
    :return: 特征值,共36个
    功能: 生成特征空间要求的特征值, 用于图像质量评分计算
    """
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 正确的转换
    img1 = img1.astype('float64')
    feat = np.array([])
    for i in range(2):  # 执行两次
        mscn_img1 = MSCN_img(img1)
        alpha, overallstd = estimate_ggdparam(mscn_img1)
        feat = np.append(feat, alpha)
        feat = np.append(feat, np.power(overallstd, 2))
        for mscn_shifted in pairs(mscn_img1):
            alpha_pair, leftstd, rightstd = estimate_aggdparam(mscn_shifted)
            const = np.divide(np.sqrt(gamma(np.divide(1, alpha_pair))),
                              np.sqrt(gamma(np.divide(3, alpha_pair))))
            meanparm = np.multiply(np.divide(gamma(np.divide(2, alpha_pair)), gamma(np.divide(1, alpha_pair))),
                                   np.multiply(np.subtract(rightstd, leftstd), const))
            feat = np.append(feat, alpha_pair)
            feat = np.append(feat, meanparm)
            feat = np.append(feat, np.power(leftstd, 2))
            feat = np.append(feat, np.power(rightstd, 2))
        # img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)  # 缩小一倍 用于多尺度处理
        img2 = Image.fromarray(img1.astype('uint8'))
        # img2 = img2.resize((109, 175),Image.ANTIALIAS)
        w, h = img2.size
        img2.thumbnail((w // 2, h // 2))  # 这种缩放默认开启抗锯齿,这步很关键!否则特征空间就全乱套了
        img1 = np.array(img2)
    return feat


def gen_test_ind(features):
    file1 = open('test_ind', mode='w')
    file1.write('1 ')
    i = 0
    for feats in features:
        i += 1
        file1.write('{0}:{1} '.format(i, feats))
    file1.close()
    pass


def compute_score(features):
    gen_test_ind(features)
    os.system('svm-scale -r allrange test_ind >> test_ind_scaled')
    os.system('svm-predict -b 1 test_ind_scaled allmodel output >>dump')
    f = open('output')
    score1 = f.readline()
    f.close()
    os.remove('output')
    os.remove('dump')
    os.remove('test_ind_scaled') # ctrl+/ 快速注释
    return score1


if __name__ == "__main__":
    # rootdir = 'D:\\graduation_design_imgfile\\simulated_image\\motion'
    # file_list = os.listdir(rootdir)
    # for i in file_list:
    #     filepath = rootdir + '\\' + i
    #     # img_load = cv2.imread("D:\\graduation_design_imgfile\\testimage1.bmp", 1)
    #     img_load = cv2.imread(filepath, 1)
    #     feats = get_feature(img_load)
    #     score = compute_score(feats)[0:4]
    #     if float(score) < 1.:
    #         score = 1.
    #     elif float(score) > 99.:
    #         score = 99.
    #     else:
    #         pass
    #     print('{0}:{1}'.format(i, score))
    img = cv2.imread('D:\\graduation_design_imgfile\\555.png', 1)
    features = get_feature(img)
    print(compute_score(features))
