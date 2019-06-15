'''
视频帧匹配脚本
'''
import numpy as np
import cv2

#至少10个点匹配
MIN_MATCH_COUNT = 10
#完全匹配偏移 d<4
BEST_DISTANCE = 4
#微量偏移  4<d<10
GOOD_DISTANCE = 10


# 特征点提取方法，内置很多种
algorithms_all = {
    "SIFT": cv2.xfeatures2d.SIFT_create(),
    "SURF": cv2.xfeatures2d.SURF_create(8000),
    "ORB": cv2.ORB_create()
}

'''
# 图像匹配
# 0完全不匹配 1场景匹配 2角度轻微偏移 3完全匹配
'''
def match2frames(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    size1 = img1.shape
    size2 = img2.shape

    img1 = cv2.resize(img1, (int(size1[1]*0.3), int(size1[0]*0.3)), cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (int(size2[1]*0.3), int(size2[0]*0.3)), cv2.INTER_LINEAR)

    sift = algorithms_all["SIFT"]

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 过滤
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) <= MIN_MATCH_COUNT:
        return 0  # 完全不匹配
    else:
        distance_sum = 0  # 特征点2d物理坐标偏移总和
        for m in good:
            distance_sum += get_distance(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt)
        distance = distance_sum / len(good)  #单个特征点2D物理位置平均偏移量

        if distance < BEST_DISTANCE:
            return 3  #完全匹配
        elif distance < GOOD_DISTANCE and distance >= BEST_DISTANCE:
            return 2  #部分偏移
        else:
            return 1  #场景匹配


'''
计算2D物理距离
'''
def get_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


if __name__ == "__main__":
    pass

'''
摄像机角度偏移告警
'''
import cv2

import numpy as np
from PIL import Image, ImageDraw, ImageFont

'''
告警信息
'''


def putText(frame, text):
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)

    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype("fonts/msyh.ttc", 30, encoding="utf-8")
    draw.text((50, 50), text, (0, 255, 255), font=font)

    cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    return cv2_text_im


texts = ["完全偏移", "严重偏移", "轻微偏移", "无偏移"]

cap = cv2.VideoCapture('D:\\graduation_design_imgfile\\555.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")

first_frame = True
pre_frame = 0

index = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if first_frame:
            pre_frame = frame
            first_frame = False
            continue

        index += 1
        if index % 24 == 0:
            result = match2frames(pre_frame, frame)
            print("检测结果===>", texts[result])

            if result > 1:  # 缓存最近无偏移的帧
                pre_frame = frame

            size = frame.shape

            if size[1] > 720:  # 缩小显示
                frame = cv2.resize(frame, (int(size[1] * 0.5), int(size[0] * 0.5)), cv2.INTER_LINEAR)

            text_frame = putText(frame, texts[result])

            cv2.imshow('Frame', text_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()