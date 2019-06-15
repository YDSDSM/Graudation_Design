import cv2
import numpy as np
from SignalLossDetection import signal_loss_decetion


def videodecode(videopath, save_path, filename1, filename2):
    """
    :param videopath: 输入视频路径+名称
    :param save_path: 保存图片的路径
    :param filename1: 保存第1帧的图片名称
    :param filename2: 保存第25帧的图片名称
    :return: None
    功能:读取视频,并保存第1帧与第25帧图片
    """
    cap = cv2.VideoCapture(videopath)
    ret, frame1 = cap.read()
    cv2.imwrite('{0}\\{1}'.format(save_path, filename1), frame1)
    for i in range(25):
        ret, frame2 = cap.read()
    ret, frame2 = cap.read()
    cv2.imwrite('{0}\\{1}'.format(save_path, filename2), frame2)
    cap.release()
    cv2.destroyAllWindows()
    return frame1, frame2


def freeze(img1, img2):
    new_img = np.abs(np.subtract(img1, img2))
    ratio, state = signal_loss_decetion(new_img)
    freeze1 = 'normal'
    if state == 'signal_loss':
        freeze1 = 'freeze'
    return freeze1


def main():
    img1, img2 = videodecode('D:\\graduation_design_imgfile\\555.mp4', 'D:\\graduation_design_imgfile',
                             '557.jpg', '558.jpg')
    str1 = freeze(img1, img2)
    print(str1)
    img3 = cv2.imread('D:\\graduation_design_imgfile\\TID2013\\distorted_images\\i01_02_1.bmp', 1)
    img4 = cv2.imread('D:\\graduation_design_imgfile\\TID2013\\distorted_images\\i01_02_1.bmp', 1)
    str2 = freeze(img3, img4)
    print(str2)


if __name__ == "__main__":
    main()
