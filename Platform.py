import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, pyqtSignal
from SharpnessAssessmentBrisqueMethod import BRISQUE_score
from BrightnessDecetion import brightness_decetion
from ColorCastDetection import colorcastdetection1
from NoiseDetection import gaussnoise
from SignalLossDetection import signal_loss_decetion
import time

class Plarform(QWidget):
    def __init__(self):  # 初始化
        super().__init__()  # 执行继承的父类的初始化办法
        self.img = np.ndarray(())
        self.openfile_name = ''
        self.flag1 = False
        self.test_degree = ['', '', '', '', '']
        self.test_score = ['', '', '', '', '']
        self.i = 0
        self.ret = False
        self.Init_UI()  # 执行Ui初始化


    def Init_UI(self):
        grid = QGridLayout()  # 创建网格布局实例
        self.setLayout(grid)  # 应用网格布局实例

        self.setGeometry(256, 256, 800, 600)  # 设置全局大小
        self.setWindowTitle('Video Superviser')  # 设置窗口名称
        # 读取图片并展示
        # img = cv2.imread('D:\\graduation_design_imgfile\\images_informal\\555.png', 1)
        # height, width, bytesPerComponent = img.shape
        # bytesPerLine = 3 * width
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.Qimg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.label_paint = QLabel()
        grid.addWidget(self.label_paint, 0, 0, 5, 5)

        # 设置按钮
        btn_openfile = QPushButton('打开文件')
        # btn_openfile.clicked.connect(self.clicked_openfile)
        grid.addWidget(btn_openfile, 6, 0)
        btn_run = QPushButton('运行')
        # btn_run.clicked.connect(self.clicked_run)
        grid.addWidget(btn_run, 6, 2)
        btn_pause = QPushButton('停止')
        grid.addWidget(btn_pause, 6, 4)
        self.label_time = QLabel('time')
        grid.addWidget(self.label_time, 6, 7)
        # 布局评价label
        rows_name = ['检测项', '评价', '评分']
        positions = [(0, i) for i in range(6, 10)]
        for position, name in zip(positions, rows_name):
            if name == '':
                continue
            label = QLabel(name)
            grid.addWidget(label, *position)

        column_names = ['清晰度', '亮度', '偏色', '信号', '噪声']
        positions = [(i, 6) for i in range(1, 6)]
        for position, name in zip(positions, column_names):
            if name == '':
                continue
            label = QLabel(name)
            grid.addWidget(label, *position)

        degrees = ['very bad', 'very bad', 'normal', 'good', 'very good']
        position = [(i, 7) for i in range(1, 6)]
        # for position, name in zip(positions, degrees):
        #     if name == '':
        #         continue
        #     label = QLabel(name)
        #     grid.addWidget(label, *position)
        self.label_ds = QLabel(degrees[0])
        self.label_ds.setToolTip("评分小于48表示图像清晰度正常")
        grid.addWidget(self.label_ds, *position[0])
        self.label_db = QLabel(degrees[1])
        self.label_db.setToolTip("评分大于20或小于80表示图像亮度正常")
        grid.addWidget(self.label_db, *position[1])
        self.label_dc = QLabel(degrees[2])
        self.label_dc.setToolTip("评分小于1表示图像偏色正常")
        grid.addWidget(self.label_dc, *position[2])
        self.label_dsi = QLabel(degrees[3])
        self.label_dsi.setToolTip("评分小于0.6表示信号未丢失")
        grid.addWidget(self.label_dsi, *position[3])
        self.label_dn = QLabel(degrees[4])
        self.label_dn.setToolTip("评分小于5表示噪声情况正常")
        grid.addWidget(self.label_dn, *position[4])

        scores = ['20', '40', '60', '80', '100']
        position = [(i, 8) for i in range(1, 6)]
        # for position, name in zip(positions, scores):
        #     if name == '':
        #         continue
        #     label = QLabel(name)
        #     grid.addWidget(label, *position)
        self.label_ss = QLabel(scores[0])
        grid.addWidget(self.label_ss, *position[0])
        self.label_sb = QLabel(scores[1])
        grid.addWidget(self.label_sb, *position[1])
        self.label_sc = QLabel(scores[2])
        grid.addWidget(self.label_sc, *position[2])
        self.label_ssi = QLabel(scores[3])
        grid.addWidget(self.label_ssi, *position[3])
        self.label_sn = QLabel(scores[4])
        grid.addWidget(self.label_sn, *position[4])

        self.thread_event = Thread1()
        btn_openfile.clicked.connect(self.clicked_openfile)
        btn_run.clicked.connect(self.clicked_run)
        btn_pause.clicked.connect(self.clicked_pause)
        self.thread_event.signal1.connect(self.updatescore)
        self.thread_event.signal2.connect(self.updatedegree)
        self.thread_event.signal3.connect(self.updatelabel)
        # self.thread_event.signal4.connect(self.updatetime)


        self.show()  # 秀出所有小器件
    def updatescore(self, score2):
        self.label_ss.setText(score2[0])
        self.label_sb.setText(score2[1])
        self.label_sc.setText(score2[2])
        self.label_ssi.setText(score2[3])
        self.label_sn.setText(score2[4])

    def updatedegree(self, degree2):
        self.label_ds.setText(degree2[0])
        self.label_db.setText(degree2[1])
        self.label_dc.setText(degree2[2])
        self.label_dsi.setText(degree2[3])
        self.label_dn.setText(degree2[4])

    def updatelabel(self, Qimg1):
        self.label_paint.setPixmap(QPixmap.fromImage(Qimg1))


    def clicked_openfile(self):
        # 定义打开文件鼠标事件
        self.openfile_name = QFileDialog.getOpenFileName(self, '选择文件', 'D:\\graduation_design_imgfile\\images_informal',
                                                    ("Video (*.png *.xpm *.jpg *.bmp *.mp4)"))
        if self.openfile_name[0]:
            self.flag1 = True

    def clicked_pause(self):
        self.thread_event.terminate()
        pass

    def clicked_run(self):
        # 定义运行鼠标事件
        if self.flag1:
            self.thread_event.path = self.openfile_name[0]
            self.thread_event.start()
            ## 这段本来是用来放视频的, 但是由于循环无法退出, 一旦运行立马无响应, 可能可以用其他方法解决问题 ↓
            # cap = cv2.VideoCapture(self.openfile_name[0])
            # self.ret, self.img = cap.read()
            # while (1):
            #     self.ret, self.img = cap.read()
            #     if self.ret:
            #         height, width, bytesPerComponent = self.img.shape
            #         bytesPerLine = 3 * width
            #         gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            #         self.Qimg = QImage(gray_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            #         self.label_paint.setPixmap(QPixmap.fromImage(self.Qimg))
            #         if self.i % 25 ==0:
            #             gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            #             self.Qimg = QImage(gray_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            #             self.label_paint.setPixmap(QPixmap.fromImage(self.Qimg))
            #             self.test_score[0], self.test_degree[0] = BRISQUE_score(self.img)
            #             self.test_score[1], self.test_degree[1] = brightness_decetion(self.img)
            #             self.test_score[2], self.test_degree[2] = colorcastdetection1(self.img)
            #             self.test_score[3], self.test_degree[3] = gaussnoise(self.img)
            #             self.test_score[4], self.test_degree[4] = signal_loss_decetion(self.img)
            #             self.label_ss.setText(self.test_score[0])
            #             self.label_ds.setText(self.test_degree[0])
            #             self.label_sb.setText(self.test_score[1])
            #             self.label_db.setText(self.test_degree[1])
            #             self.label_sc.setText(self.test_score[2])
            #             self.label_dc.setText(self.test_degree[2])
            #             self.label_ssi.setText(self.test_score[3])
            #             self.label_dsi.setText(self.test_degree[3])
            #             self.label_sn.setText(self.test_score[4])
            #             self.label_dn.setText(self.test_degree[4])
            #         self.i += 1
            ## ↑↑ 未完成代码




class Thread1(QThread):
    signal1 = pyqtSignal(list)
    signal2 = pyqtSignal(list)
    signal3 = pyqtSignal(QImage)
    # signal4 = pyqtSignal(float)

    def __init__(self):
        QThread.__init__(self)
        self.test_degree = ['', '', '', '', '']
        self.test_score = ['', '', '', '', '']
        self.path = ''
        self.i = 0
        # self.time_start = 0.0
        # self.elasped_time = 0.0

    def __del__(self):
        self.quit()
        self.wait()


    def run(self):
        if self.path != '':
            cap = cv2.VideoCapture(self.path)
            self.ret, self.img = cap.read()
            height, width, bytesPerComponent = self.img.shape
            bytesPerLine = 3 * width
            while (1):
                self.ret, self.img = cap.read()
                if self.ret:
                    gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                    self.Qimg = QImage(gray_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    self.signal3.emit(self.Qimg)
                    # Plarform.label_paint.setPixmap(QPixmap.fromImage(self.Qimg))
                    if self.i % 80 == 0:
                        # gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                        # self.Qimg = QImage(gray_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        # self.label_paint.setPixmap(QPixmap.fromImage(self.Qimg))
                        # self.time_start = time.perf_counter()
                        self.test_score[0], self.test_degree[0] = BRISQUE_score(self.img)
                        self.test_score[1], self.test_degree[1] = brightness_decetion(self.img)
                        self.test_score[2], self.test_degree[2] = colorcastdetection1(self.img)
                        self.test_score[3], self.test_degree[3] = gaussnoise(self.img)
                        self.test_score[4], self.test_degree[4] = signal_loss_decetion(self.img)
                        self.signal1.emit(self.test_score)
                        self.signal2.emit(self.test_degree)
                        # self.elasped_time = time.perf_counter() - self.time_start
                        # self.signal4.emit(self.elasped_time)
                        # Plarform.label_ss.setText(self.test_score[0])
                        # Plarform.label_ds.setText(self.test_degree[0])
                        # Plarform.label_sb.setText(self.test_score[1])
                        # Plarform.label_db.setText(self.test_degree[1])
                        # Plarform.label_sc.setText(self.test_score[2])
                        # Plarform.label_dc.setText(self.test_degree[2])
                        # Plarform.label_ssi.setText(self.test_score[3])
                        # Plarform.label_dsi.setText(self.test_degree[3])
                        # Plarform.label_sn.setText(self.test_score[4])
                        # Plarform.label_dn.setText(self.test_degree[4])
                    self.i += 1
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Plarform()
    app.exit(app.exec_())
