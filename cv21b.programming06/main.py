import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


class MessageItem(object):
    # 用于封装信息的类,包含图片和其他信息
    def __init__(self, frame, message):
        self._frame = frame
        self._message = message

    def getFrame(self):
        # 图片信息
        return self._frame

    def getMessage(self):
        # 文字信息,json格式
        return self._message


class Tracker(object):
    '''
    追踪者模块,用于追踪指定目标
    '''

    def __init__(self, tracker_type="BOOSTING", draw_coord=True):
        '''
        初始化追踪器种类
        '''
        # 获得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        # 构造追踪器
        if int(major_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()

    def initWorking(self, frame, box):
        '''
        追踪器工作初始化
        frame:初始化追踪画面
        box:追踪的区域
        '''
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame, box)
        # if not status:
        #     raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        '''
        开启追踪
        '''
        message = None
        p1, p2 = (), ()
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            p1 = (int(self.coord[0]), int(self.coord[1]))
            p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
            if status:
                message = {"coord": [((int(self.coord[0]), int(self.coord[1])),
                                      (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
                if self.draw_coord:
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    message['msg'] = "is tracking"
        return p1, p2


if __name__ == '__main__':
    path = "dataset\\test_public\\test_public"
    # 若要跟踪训练验证集
    # path = "dataset\\trainval\\trainval"
    count = 0
    for dire in tqdm(os.listdir(path)):
        im = Image.open(os.path.join(path, dire, "00000001.jpg"))
        img = np.asarray(im, dtype=np.int32)
        gt = open(os.path.join(path, dire, "groundtruth.txt"),
                  "r").readlines()
        c = gt[0].strip("\n").split(",")
        # c = "923.65,308.27,758.05,342.11,741.83,262.68,907.42,228.85".split(",")
        # c = "741.83,228.85,923.65,228.85,923.65,342.11,741.83,342.11".split(",")
        x1, y1 = int(float(c[0])), int(float(c[1]))
        w = abs(int(float(c[4])) - x1)
        h = int(float(c[5])) - y1

        roi = (x1, y1, w, h)
        # print(roi)
        #
        # # roi 初始目标位置
        # # img 初始图像
        #
        gTracker = Tracker(tracker_type="KCF")
        gTracker.initWorking(img, roi)
        #
        # # 循环帧读取，开始跟踪
        file = os.listdir(os.path.join(path, dire))
        lt_cache, rb_cache = (0, 0), (0, 0)
        frame_cache = 0
        res = gt[0]
        for f in file:
            if "txt" in f or "00000001.jpg" in f:
                continue
            name = os.path.join(path, dire, f)
            # print(f)
            frame = np.asarray(Image.open(name))
            # print(frame.shape)
            lt, rb = gTracker.track(frame)
            # if lose the target, use the latest frame
            if lt == (0, 0) or rb == (0, 0):
                # print(dire+" " + f)
                gTracker = Tracker(tracker_type="KCF")
                roi_u = (lt_cache[0], lt_cache[1], rb_cache[0]-lt_cache[0], rb_cache[1]-lt_cache[1])
                gTracker.initWorking(frame_cache, roi_u)
                lt, rb = gTracker.track(frame)
                if lt == (0, 0) or rb == (0, 0):
                    print(dire + " " + f+" fuck")
                    break
            else:
                lt_cache, rb_cache = lt, rb
                frame_cache = frame
            res = res + \
                  str(lt[0]) + "," + str(lt[1]) + "," +\
                  str(rb[0]) + "," + str(lt[1]) + "," +\
                  str(rb[0]) + "," + str(rb[1]) + "," +\
                  str(lt[0]) + "," + str(rb[1]) + "\n"
        with open(dire+".txt", "a") as r:
            r.write(res)

