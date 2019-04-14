import cv2
import numpy as np

"""
本段程序实现了信号丢失检测, 函数返回最大连通分量面积比,以及最大连通分量图,具体使用方法如下
img_load = cv2.imread("D:\\graduation_design_imgfile\\signal_loss.jpg", 1)
ratio, img2 = signal_loss_decetion(img_load)
if ratio >0.9:
    print('信号丢失')
下面是一些cv2.threshold使用例子,留给自己以后看的
# ret, th3 = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)  # 自定义阈值法
# # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 大津阈值法
# # cv2.imshow('img_bin', img_binary)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
"""


def signal_loss_decetion(img):
    """
    :param img: 输入图像,彩图
    :return: 最大连通分量面积/图片面积, 最大连通分量图
    功能,通过观察最大连通分量面积与图片面积的比值,判断信号是否丢失;方法生效的前提是基于信号丢失时呈现的图片基本是黑的这一事实.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mean = np.mean(img)
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    S_ratio = 0
    img_maxtype = np.zeros(img.shape)
    if img_mean < 45:
        image = img_bin.astype('uint8')
        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)  # 参数说明:
        # @ param: nb_components  连通分量数
        # @ param: output 带标记的输出矩阵, 具体来说就是连通分量1在输出里被标记为1, 连通分量2在输出里被标记为2.
        # @ param: stats  输出为大小是 nb_components X 5的矩阵, 每一列的参数说明如下,
        # 第一列,最左(x)的坐标 第二列,最上(y)的坐标(前两列可以用于定位 第三列,水平宽度,第四列,垂直宽度,第五列,总面积
        # @ param: centroids 种子原点
        sizes = stats[:, -1]  # 获取stats最后一列的信息,该信息记录了连通分量的面积
        max_label = 0  # 设定初始值
        max_size = sizes[0]  # 设定初始值
        cons = int(np.divide(img.size, 2))  # 为退出循环做准备,即当最大连通分量面积超过图像面积的0.5倍时,就可以退出循环了.
        for i in range(1, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
            elif max_size > cons:
                break
        S_ratio = np.divide(max_size, img.size)
        img_maxtype[labels == max_label] = 255
    return S_ratio, img_maxtype


if __name__ == "__main__":
    img_load = cv2.imread("D:\\graduation_design_imgfile\\signal_loss.jpg", 1)
    ratio, img2 = signal_loss_decetion(img_load)
    print(ratio)
    if ratio >0.9:
        print('信号丢失')
    # plt.imshow(img2, 'gray')
    # plt.show()
