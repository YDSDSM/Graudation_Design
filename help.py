# 利用ID2013数据库检测算法性能
import cv2
import os
import time
from NoiseDetection import gaussnoise
from BrightnessDecetion import brightness_decetion
from SharpnessAssessmentBrisqueMethod import BRISQUE_score
time_start = time.perf_counter()
# cap = cv2.VideoCapture('D:\\graduation_design_imgfile\\test.mp4')
# ret, frame = cap.read()
# i = 0
# while(1):
#     ret, frame = cap.read()
#     if ret:
#         if i % 25 == 0:
#             sigma = gaussnoise(frame)
#             print('frame{0}| sigma {1}'.format(i, sigma))
#         i += 1
#         cv2.imshow('img2', frame)
#         k = cv2.waitKey(60) & 0xff
#         if k == 27:
#             break
#         else:
#             pass
#     else:
#         break
rootdir = 'D:\\graduation_design_imgfile\\TID2013\\distorted_images'
file_list = os.listdir(rootdir)
# file_list.remove('Thumbs.db')
mos_score = []
file1 = open('D:\\graduation_design_imgfile\\TID2013\\mos.txt')
for line in file1.readlines():
    mos_score.append(line)
# distort_num = ['08', '09']
# degree_num = ['4', '5']
#
# for i in range(len(file_list)):
#     if i % 120 == 1 or i % 120 == 2 or i % 120 ==3 or i % 120 == 4 or i % 120 == 0:
#         img_name = file_list[i]
#         img_load = cv2.imread('{0}\\{1}'.format(rootdir, img_name), 1)
#         sigma = gaussnoise(img_load)
#         txt = "{0}|{1}".format(img_name, sigma)
#         print(txt)
print('img_name|socre|degree|mos')
k = 0
for i in file_list:
    img_name = '{0}\\{1}'.format(rootdir, i)
    img_load = cv2.imread(img_name, 1)
    score, degree = BRISQUE_score(img_load)
    txt = "{0}|{1}|{2}|{3}".format(i, score, degree, mos_score[k])
    print(txt)
    k += 1
time_elasped = time.perf_counter() - time_start
print('elasped {0} secend'.format(time_elasped))

