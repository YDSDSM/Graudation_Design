import cv2
import numpy as np

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
        print('dark')
    elif brightness_score > 80:
        print('Very bright')
    else:
        print('Normal')

    return brightness_score


if __name__ == "__main__":
    img = cv2.imread('D:\\graduation_design_imgfile\\signal_loss.jpg')
    print(brightness_decetion(img))
