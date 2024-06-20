import argparse
import os
import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device
import warnings
from collections import deque
from bot_tracker.mc_bot_sort import BoTSORT
import math
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# 检测参数
parser.add_argument('--weights', default=r"./weights/yolov5s.pt", type=str, help='weights path')
parser.add_argument('--source', default=r"./vedio/night_0101.mp4", type=str, help='img or video(.mp4)path')
parser.add_argument('--save', default=r"./data/save", type=str, help='save img or video path')
parser.add_argument('--vis', default=True, action='store_true', help='visualize image')
parser.add_argument('--device', type=str, default="0", help='use gpu or cpu')
parser.add_argument('--imgsz', type=tuple, default=(640, 640), help='image size')
parser.add_argument('--merge_nms', default=False, action='store_true', help='merge class')
parser.add_argument('--conf_thre', type=float, default=0.5, help='conf_thre')
parser.add_argument('--iou_thre', type=float, default=0.5, help='iou_thre')

# 跟踪参数
parser.add_argument('--track_high_thresh', type=float, default=0.3, help='track_high_thresh')
parser.add_argument('--track_low_thresh', type=float, default=0.05, help='track_low_thresh')
parser.add_argument('--new_track_thresh', type=float, default=0.4, help='new_track_thresh')
parser.add_argument('--match_thresh', type=float, default=0.7, help='match_thresh')
parser.add_argument('--track_buffer', type=int, default=30, help='track_buffer')
parser.add_argument('--frame_rate', type=int, default=30, help='frame_rate')
parser.add_argument('--proximity_thresh', type=float, default=0.5, help='proximity_thresh')
parser.add_argument('--appearance_thresh', type=float, default=0.25, help='appearance_thresh')

# 添加高度（H）、角度（alpha）和相机内参（calib）参数
parser.add_argument('--H', type=float, default=6.5, help='Height of the camera from the ground')
parser.add_argument('--alpha', type=int, default=8, help='Angle of the camera view')
parser.add_argument('--calib', type=str, default='2900,0.0,640; 0.0,2900,360; 0.0,0.0,1.0',
                    help='Camera intrinsic parameters')
# 解析参数
opt = parser.parse_args()
# 将字符串形式的 calib 转换为 numpy 数组
calib_values = [float(x) for x in opt.calib.replace(';', ',').split(',')]
opt.calib = np.array(calib_values).reshape(3, 3)


def draw_measure_line(H, calib, u, v, alpha):
    alpha = alpha  # 角度a

    # 相机焦距
    fy = calib[1][1]
    # 相机光心
    u0 = calib[0][2]
    v0 = calib[1][2]

    pi = math.pi

    Q_pie = [u - u0, v - v0]
    gamma_pie = math.atan(Q_pie[1] / fy) * 180 / np.pi

    beta_pie = alpha + gamma_pie

    if beta_pie == 0:
        beta_pie = 1e-2

    z_in_cam = (H / math.sin(beta_pie / 180 * pi)) * math.cos(gamma_pie * pi / 180)

    return z_in_cam

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Detector:
    def __init__(self, device, origin_imgsz, model_path=r'./best_dist_model.pt', imgsz=(640, 640), conf=0.5, iou=0.0625,
                 merge_nms=False):

        # video1
        # self.pointlist = [(1005, 720), (1005, 520)]
        # video2
        # self.pointlist = [(221, 538), (740, 538)]
        # self.pointlist = [(6, 410), (1278, 410)]
        self.pointlist = [(0, int(origin_imgsz[1]/2)), (size[0], int(origin_imgsz[1]/2))]

        self.device = device
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False)
        self.names = self.model.names

        self.stride = self.model.stride

        self.imgsz = check_img_size(imgsz, s=self.stride)

        self.conf = conf

        self.iou = iou

        self.merge_nms = merge_nms

        self.tracker = BoTSORT(track_high_thresh=opt.track_high_thresh, track_low_thresh=opt.track_low_thresh,
                               new_track_thresh=opt.new_track_thresh,
                               match_thresh=opt.match_thresh, track_buffer=opt.track_buffer, frame_rate=opt.frame_rate,
                               with_reid=False, proximity_thresh=False, appearance_thresh=False,
                               fast_reid_config=None, fast_reid_weights=None, device=None)

        self.trajectories = {}

        self.max_trajectory_length = 5

        self.id_in = 0
        self.id_in_list = []

        self.id_out = 0
        self.id_out_list = []

    @torch.no_grad()
    def __call__(self, image: np.ndarray):
        img_vis = image.copy()
        img = letterbox(image, self.imgsz, stride=self.stride)[0]
        # print(img.shape)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        im = img.float()  # uint8 to fp16/32
        im /= 255.0
        im = im[None]
        # inference
        pred = self.model(im, augment=False, visualize=False)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.conf, iou_thres=self.iou, classes=None,
                                   agnostic=self.merge_nms, max_det=1000)

        # 划中心线
        # cv2.line(img_vis, self.pointlist[0], self.pointlist[1], (0, 255, 0), 2)


        for i, det in enumerate(pred):  # detections per image
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()
            online_targets = self.tracker.update(det[:, :6].cpu(), image)
            for t in online_targets:
                speed_km_per_h = 0
                tlwh = t.tlwh
                tid = t.track_id
                cls = int(t.cls.item())
                color = get_color(int(cls) + 2)
                # center = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2))
                bottom_center = (int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3]))
                cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])),
                              (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), color, 2)

                zc_cam_other = draw_measure_line(opt.H, opt.calib, int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3]), opt.alpha)

                # if self.names[int(cls)] == "person":
                if tid not in self.trajectories:
                    self.trajectories[tid] = deque(maxlen=self.max_trajectory_length)

                trajectory_point = {
                    'bottom_center': bottom_center,
                    'zc_cam_other': zc_cam_other
                }
                self.trajectories[tid].appendleft(trajectory_point)

                # 截断轨迹长度
                if len(self.trajectories[tid]) > self.max_trajectory_length:
                    self.trajectories[tid] = self.trajectories[tid][:self.max_trajectory_length]

                for i in range(1, len(self.trajectories[tid])):

                    if self.trajectories[tid][i - 1] is None or self.trajectories[tid][i] is None:
                        continue

                    thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    cv2.line(img_vis, self.trajectories[tid][i - 1]['bottom_center'],
                             self.trajectories[tid][i]['bottom_center'], color,
                             thickness)

                    # 通过阈值判断是竖线还是横线,若x2和x1之间的差值小于20 判断为竖线 否则为横线
                    """
                    例如：如下垂线上下两点的x保持不变,y存在变化 所以垂线的x2-x1=0
                        *
                        *
                        *
                        *

                        如下横线左右两点的y保持不变,x存在变化 所以横线的y2-y1=0
                        ***********
                    """
                    if abs(self.pointlist[1][0] - self.pointlist[0][0]) < 20:
                        # 通过判断同一个目标上下两帧是否过线 来进行计数
                        # self.trajectories[tid][i - 1][0]代表上一帧 self.trajectories[tid][i][0]代表当前帧
                        if ((self.trajectories[tid][i - 1]['bottom_center'][0] <= self.pointlist[0][0]) and
                            (self.trajectories[tid][i]['bottom_center'][0] > self.pointlist[0][0])) and \
                                ((self.trajectories[tid][i]['bottom_center'][1] > self.pointlist[1][
                                    1]) and  # 设置目标的撞线范围 不得超线
                                 (self.trajectories[tid][i]['bottom_center'][1] < self.pointlist[0][1])):

                            # 如果目标ID已经计数过，则忽略
                            if tid in self.id_in_list:
                                continue
                            # 否则，增加进入计数，并将ID添加到已计数列表中
                            else:
                                self.id_in += 1
                                self.id_in_list.append(tid)

                        if ((self.trajectories[tid][i - 1]['bottom_center'][0] >= self.pointlist[0][0]) and
                            (self.trajectories[tid][i]['bottom_center'][0] < self.pointlist[0][0])) and \
                                ((self.trajectories[tid][i]['bottom_center'][1] > self.pointlist[1][1]) and
                                 (self.trajectories[tid][i]['bottom_center'][1] < self.pointlist[0][1])):

                            if tid in self.id_out_list:
                                continue
                            else:
                                self.id_out += 1
                                self.id_out_list.append(tid)

                    else:
                        if ((self.trajectories[tid][i - 1]['bottom_center'][1] >= self.pointlist[0][1]) and
                            (self.trajectories[tid][i]['bottom_center'][1] < self.pointlist[0][1])) and \
                                ((self.trajectories[tid][i]['bottom_center'][0] > self.pointlist[0][0]) and
                                 (self.trajectories[tid][i]['bottom_center'][0] < self.pointlist[1][0])):

                            if tid in self.id_in_list:
                                continue
                            else:
                                self.id_in += 1
                                self.id_in_list.append(tid)

                        if ((self.trajectories[tid][i - 1]['bottom_center'][1] <= self.pointlist[0][1]) and
                            (self.trajectories[tid][i]['bottom_center'][1] > self.pointlist[0][1])) and \
                                ((self.trajectories[tid][i]['bottom_center'][0] > self.pointlist[0][0]) and
                                 (self.trajectories[tid][i]['bottom_center'][0] < self.pointlist[1][0])):

                            if tid in self.id_out_list:
                                continue
                            else:
                                self.id_out += 1
                                self.id_out_list.append(tid)

                    time_interval = 1 / fps
                    speed_m_per_s = abs(self.trajectories[tid][i]['zc_cam_other'] - self.trajectories[tid][i - 1][
                        'zc_cam_other']) / time_interval
                    speed_km_per_h = speed_m_per_s * 3.6  # 转换为公里/小时

                # 显示类名和跟踪ID
                cv2.putText(img_vis, f"{self.names[int(cls)]} {int(tid)}",
                            (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

                # 显示速度，纵坐标比类名和跟踪ID的位置稍低一些
                cv2.putText(img_vis, f"{speed_km_per_h:.1f} km/h",
                            (int(tlwh[0]), int(tlwh[1]) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

                cv2.putText(img_vis, f"{zc_cam_other:.1f} M",
                            (int(tlwh[0]), int(tlwh[1]) + 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

                # img_vis = cv2ImgAddText(img_vis, str(f"下行车辆数量:{self.id_in}"), int(img_vis.shape[1] * 0.1), int(img_vis.shape[0] * 0.17))
                # img_vis = cv2ImgAddText(img_vis, str(f"上行车辆数量:{self.id_out}"), int(img_vis.shape[1] * 0.1), int(img_vis.shape[0] * 0.1))

                # cv2.putText(img_vis, str(f"下行车辆数量:{self.id_in}"),
                #             (int(img_vis.shape[1] * 0.1), int(img_vis.shape[0] * 0.17)), cv2.FONT_HERSHEY_COMPLEX,
                #             2,
                #             (0, 255, 0), 2)

                # cv2.putText(img_vis, str(f"上行车辆数量:{self.id_out}"),
                #             (int(img_vis.shape[1] * 0.1), int(img_vis.shape[0] * 0.1)), cv2.FONT_HERSHEY_COMPLEX,
                #             2,
                #             (0, 0, 255), 2)


        return img_vis


if __name__ == '__main__':
    print("开始生成数据")
    device = select_device(opt.device)
    print(device)


    capture = cv2.VideoCapture(opt.source)
    frame_id = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    outVideo = cv2.VideoWriter(os.path.join(opt.save, os.path.basename(opt.source).split('.')[-2] + "_out.mp4"), fourcc,
                               fps, size)
    # print("size", size)

    model = Detector(device=device, model_path=opt.weights, origin_imgsz=size, imgsz=opt.imgsz, conf=opt.conf_thre,
                     iou=opt.iou_thre,
                     merge_nms=opt.merge_nms)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        img_vis = model(frame)
        outVideo.write(img_vis)
        frame_id += 1

        img_vis = cv2.resize(img_vis, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('track', img_vis)
        cv2.waitKey(30)

    capture.release()
    outVideo.release()
