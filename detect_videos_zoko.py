# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import pytz
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

import winsound
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from shapely.geometry import Polygon, LineString, Point
from csv import writer
from datetime import datetime, timedelta
# set timezone
as_TKY = pytz.timezone('Asia/Tokyo')
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


@dataclass
class CameraParameters:
    focal_length_mm: float
    pixel_size_mm: float
    helmet_height_mm: float
    hook_height_mm: float
    camera_height_m: float


@dataclass
class HeightEstimationResult:
    hook_distance_m: float
    helmet_distance_m: float
    hook_height_m: float
    helmet_height_m: float
    relative_height_m: float


def estimate_distance_m(pixel_height: float, real_height_mm: float, params: CameraParameters) -> Optional[float]:
    """Estimate object distance from the camera along the optical axis."""

    if pixel_height <= 0 or real_height_mm <= 0 or params.pixel_size_mm <= 0:
        return None

    distance_mm = (real_height_mm * params.focal_length_mm) / (params.pixel_size_mm * pixel_height)
    return distance_mm / 1000.0


def estimate_height_metrics(helmet_px: float, hook_px: float, params: CameraParameters) -> Optional[HeightEstimationResult]:
    """Compute height-related metrics for the crane hook and helmet."""

    hook_distance_m = estimate_distance_m(hook_px, params.hook_height_mm, params)
    helmet_distance_m = estimate_distance_m(helmet_px, params.helmet_height_mm, params)

    if hook_distance_m is None or helmet_distance_m is None:
        return None

    hook_height_m = max(params.camera_height_m - hook_distance_m, 0.0)
    helmet_height_m = max(params.camera_height_m - helmet_distance_m, 0.0)
    relative_height_m = helmet_distance_m - hook_distance_m

    return HeightEstimationResult(
        hook_distance_m=hook_distance_m,
        helmet_distance_m=helmet_distance_m,
        hook_height_m=hook_height_m,
        helmet_height_m=helmet_height_m,
        relative_height_m=relative_height_m,
    )


def select_reference_helmet(helmet_entry: Dict[str, List[List[float]]]) -> Tuple[Optional[List[float]], str]:
    """Select the representative helmet detection for distance measurements."""

    if not helmet_entry:
        return None, ''

    combined: List[Tuple[List[float], str]] = []
    for label in ('helmet', 'cross'):
        for detection in helmet_entry.get(label, []):
            combined.append((detection, label))

    if not combined:
        return None, ''

    combined.sort(key=lambda item: max(item[0][6], item[0][7]), reverse=True)
    detection, label = combined[0]
    return detection, label


def compute_horizontal_distance_m(
    crane_entry: Optional[List[float]],
    helmet_entry: Optional[List[float]],
    params: CameraParameters,
    frame_shape: Tuple[int, int, int],
) -> Optional[float]:
    """Compute the horizontal distance between the crane hook and helmet in meters."""

    if not crane_entry or not helmet_entry:
        return None

    frame_height, frame_width = frame_shape[0], frame_shape[1]
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    hook_size = max(crane_entry[6], crane_entry[7])
    helmet_size = max(helmet_entry[6], helmet_entry[7])

    if hook_size <= 0 or helmet_size <= 0:
        return None

    hook_offset_x = crane_entry[4] - center_x
    hook_offset_y = crane_entry[5] - center_y
    helmet_offset_x = helmet_entry[4] - center_x
    helmet_offset_y = helmet_entry[5] - center_y

    hook_lx_mm = params.hook_height_mm * hook_offset_x / hook_size
    hook_ly_mm = params.hook_height_mm * hook_offset_y / hook_size
    helmet_lx_mm = params.helmet_height_mm * helmet_offset_x / helmet_size
    helmet_ly_mm = params.helmet_height_mm * helmet_offset_y / helmet_size

    horizontal_mm = math.hypot(hook_lx_mm - helmet_lx_mm, hook_ly_mm - helmet_ly_mm)
    return horizontal_mm / 1000.0

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# set warining sound params (in msec)
stime1 = 0
stime2 = 0
warn1_len = 3437
warn2_len = 3135

def display_warning(frame, warning_type, mute):
    global stime1, stime2
    global warn1_len, warn2_len
    if warning_type == 1:
        s_img = cv2.imread("warnings/warning1.png", cv2.IMREAD_UNCHANGED)
        x_offset=75
        y_offset=75
        frame[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
        ctime1 = int(time.time() * 1000)
        if ctime1 - stime1 > warn1_len:
            if not mute:
                winsound.PlaySound("warnings/warning1a.wav", winsound.SND_FILENAME | winsound.SND_ASYNC )
            stime1 = ctime1
    elif warning_type == 2:
        s_img = cv2.imread("warnings/warning2.png", cv2.IMREAD_UNCHANGED)
        x_offset=75
        y_offset=275
        frame[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
        ctime2 = int(time.time() * 1000)
        if ctime2 - stime2 > warn2_len:
            if not mute:
                winsound.PlaySound("warnings/warning2a.wav", winsound.SND_FILENAME | winsound.SND_ASYNC )
            stime2 = ctime2
    else:
        frame = frame
    return frame


def display_zoom_warning(frame, warning_type):
    if warning_type == 1:
        s_img = cv2.imread("warnings/zoom_up.png", cv2.IMREAD_UNCHANGED)
        x_offset=25
        y_offset=600
        frame[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    elif warning_type == 2:
        s_img = cv2.imread("warnings/zoom_down.png", cv2.IMREAD_UNCHANGED)
        x_offset=25
        y_offset=650
        frame[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    else:
        frame = frame
    return frame

def calc_queue_avg(queue, m):
    queue_non_zero = list(filter(None,queue))
    if len(queue_non_zero) >= m and queue:
        return round(sum(queue_non_zero) / len(queue_non_zero))
    else:
        return 0

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        skip_step=1, # inference every nth frame
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        height_diff_thres=4.2, # height difference threshold
        crane_width_queue_args = [10,2],
        helmet_width_queue_args = [10,2],
        helmet_detect_queue_args = [10,1],
        focal_length_mm=4.0,
        pixel_size_mm=0.00112,
        helmet_height_mm=240.0,
        hook_height_mm=300.0,
        camera_height_m=10.0,
        min_zoom_level = 10, # minimum camera zoom level
        max_zoom_level = 25, # maximum camera zoom level
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_csv=False, # save results to *.csv
        mute_warnings=False, # mute warning sounds
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    stime1 = 0
    stime2 = 0
    warn1_len = 7784
    warn2_len = 9195
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, skip_step=skip_step)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    if webcam:
        prev_im = [None] * bs

    # initialize camera parameters
    camera_params = CameraParameters(
        focal_length_mm=focal_length_mm,
        pixel_size_mm=pixel_size_mm,
        helmet_height_mm=helmet_height_mm,
        hook_height_mm=hook_height_mm,
        camera_height_m=camera_height_m,
    )

    # initialize queues
    helmet_height_queue = Queue()
    crane_height_queue = Queue()
    helmet_detect_queue = Queue()

    # initialize queue parameters
    h_k, h_m = helmet_width_queue_args
    c_k, c_m = crane_width_queue_args

    hd_k, hd_m = helmet_detect_queue_args

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        if webcam:
            # Optimize repeated img
            if (prev_im == im).all():
                continue
            prev_im = im

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            csv_path =  str(save_dir / p.stem)  # im.csv
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if not webcam:
                curr_time = datetime.fromtimestamp(frame/30).strftime('%H:%M:%S.%f')
                cv2.putText(im0, 'Frame: {0}'.format(frame), ((im0.shape[1]-300), 50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                curr_time = datetime.now(as_TKY).strftime("%Y-%m-%d %H:%M:%S.%f")
                curr_time_vis = datetime.now(as_TKY).strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(im0, 'Time: {0}'.format(curr_time_vis), ((im0.shape[1]-500), 50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 255, 255), 2, cv2.LINE_AA)
                
            # Dequeue queues if k limit is reached
            if crane_height_queue.size() == c_k and crane_height_queue.size() != 0:
                crane_height_queue.dequeue()

            if helmet_height_queue.size() == h_k and helmet_height_queue.size() != 0:
                helmet_height_queue.dequeue()
            
            if helmet_detect_queue.size() == hd_k and helmet_detect_queue.size() != 0:
                helmet_detect_queue.dequeue()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                det_dict = {0:[],1:[],2:[],3:[],4:[]}

                warning = 0
                zoom_warning = 0
                crane_ratio = 0.0
                height_difference_m = 0.0
                hook_distance_m = 0.0
                helmet_distance_m = 0.0
                relative_height_m = 0.0
                relative_horizontal_m = 0.0
                
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    c = int(cls)
                    obj = names[c]
                    data = [
                        frame,
                        curr_time,
                        obj,
                        float(conf),
                        x_c,
                        y_c,
                        bbox_w,
                        bbox_h,
                        0.0,
                        warning,
                        '',
                        '',
                        '',
                        '',
                        '',
                        '',
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

                    # append data to detection dictionary
                    if c not in det_dict:
                        det_dict[c] = []
                    det_dict[c].append(data)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
                # Add helmet and crane heights to queues if exists
                if det_dict[3]: # crane entry
                    det_dict[3] = sorted(det_dict[3], key=lambda x: x[3])[::-1] # sort crane hooks by highest conf level
                    crane = det_dict[3][0]
                    crane_height_queue.enqueue(crane[7])
                else:
                    crane_height_queue.enqueue(0)

                helmets = det_dict[1] + det_dict[4] # combine helmets detections


                if helmets: # check for helmet detections
                    helmets = sorted(helmets, key=lambda x: x[7], reverse=True) # sort helmets by height
                    helmet_height_queue.enqueue(helmets[0][7])
                    helmet_detect_queue.enqueue({'helmet': det_dict[1], 'cross': det_dict[4]}) # add detections to queue
                else:
                    helmet_height_queue.enqueue(0)
                    helmet_detect_queue.enqueue({}) # add detections to queue

                # Calculated average heights
                avg_crane_height = calc_queue_avg(crane_height_queue.items, c_m)
                avg_helmet_height = calc_queue_avg(helmet_height_queue.items, h_m)

                # Determine zoom warning
                if avg_helmet_height > 0 and avg_helmet_height <= min_zoom_level:
                    zoom_warning = 1
                elif avg_helmet_height >= max_zoom_level:
                    zoom_warning = 2
                else:
                    zoom_warning = 0

                # Calculate HD
                helmet_detect_binary = [1 if l else 0 for l in helmet_detect_queue.items]
                helmet_det_cnt = helmet_detect_binary.count(1) # number of detections last 10 frames

                if helmet_det_cnt >= hd_m:
                    helmet_detect = 1
                else:
                    helmet_detect = 0
                    
                latest_helmet = None
                reference_detection = None

                if avg_crane_height > 0 and avg_helmet_height > 0:  # check both crane and helmet values exist

                    height_result = estimate_height_metrics(avg_helmet_height, avg_crane_height, camera_params)

                    if height_result:
                        crane_ratio = avg_crane_height / avg_helmet_height if avg_helmet_height > 0 else 0.0
                        relative_height_m = height_result.relative_height_m
                        height_difference_m = abs(relative_height_m)
                        hook_distance_m = height_result.hook_distance_m
                        helmet_distance_m = height_result.helmet_distance_m
                    else:
                        crane_ratio = 0.0
                        height_difference_m = 0.0
                        relative_height_m = 0.0
                        hook_distance_m = 0.0
                        helmet_distance_m = 0.0

                    if bool(helmet_detect):  # check helmet detection in last n detections

                        helmet_index = np.min(np.nonzero(helmet_detect_binary))  # latest detection index
                        latest_helmet = helmet_detect_queue.items[helmet_index]
                        reference_detection, _ = select_reference_helmet(latest_helmet)

                        horizontal_distance = compute_horizontal_distance_m(
                            crane,
                            reference_detection,
                            camera_params,
                            im0.shape,
                        )

                        if horizontal_distance is not None:
                            relative_horizontal_m = horizontal_distance

                    # Calculate warnings based on height difference threshold
                    if height_difference_m > height_diff_thres:

                        list_cross_helmets = [a_dict['cross'] for a_dict in helmet_detect_queue.items if a_dict]
                        cross_helmet_bool = any(list_cross_helmets)  # check if any cross_helmets in queue

                        if (
                            bool(helmet_detect)
                            and latest_helmet
                            and not cross_helmet_bool
                            and latest_helmet.get('helmet')
                        ):  # if no cross helmet in image while normal helmet in image
                            warning = 2

                        if warning < 2 and bool(helmet_detect) and latest_helmet:
                            # make buffer
                            crane_buf = Point(int(crane[4]), int(crane[5])).buffer(10 * avg_helmet_height)

                            # draw buffer red
                            cv2.circle(
                                im0,
                                (int(crane[4]), int(crane[5])),
                                (10 * avg_helmet_height),
                                (0, 0, 255),
                                2,
                            )

                            for helmet in latest_helmet.get('helmet', []) + latest_helmet.get('cross', []):  # helmet detections
                                if crane_buf.contains(Point(helmet[4], helmet[5])):  # if any helmet in buffer
                                    warning = 1
                                    break
                    else:
                        warning = 0
                else:
                    crane_ratio = 0.0
                    height_difference_m = 0.0
                    relative_height_m = 0.0
                    hook_distance_m = 0.0
                    helmet_distance_m = 0.0

                # Write vars to crane entry
                if det_dict[3]: # only write analysis vars if crane exists

                    # Crane/helmet ratio (legacy log column)
                    det_dict[3][0][8] = crane_ratio

                    # Warnings
                    det_dict[3][0][9] = warning

                    # Object size lists
                    det_dict[3][0][10] = crane_height_queue.items
                    det_dict[3][0][11] = helmet_height_queue.items

                    # Object detection lists
                    det_dict[3][0][12] = helmet_detect_binary

                    # Avg size values
                    det_dict[3][0][13] = avg_crane_height
                    det_dict[3][0][14] = avg_helmet_height
                    det_dict[3][0][15] = helmet_detect
                    det_dict[3][0][16] = height_difference_m
                    det_dict[3][0][17] = hook_distance_m
                    det_dict[3][0][18] = helmet_distance_m
                    det_dict[3][0][19] = relative_height_m
                    det_dict[3][0][20] = relative_horizontal_m

                if save_csv: # Write to CSV
                    # write the header if not exists
                    if not (os.path.exists(csv_path + '.csv')):
                        header = [
                            'frame',
                            'time',
                            'class',
                            'confidence',
                            'x_local',
                            'y_local',
                            'box_width',
                            'box_height',
                            'crane_helmet_ratio',
                            'crane_warning',
                            'crane_size_list',
                            'helmet_size_list',
                            'helmet_detect_list',
                            'avg_crane_width',
                            'avg_helmet_width',
                            'helmet_detect',
                            'height_difference_m',
                            'hook_distance_m',
                            'helmet_distance_m',
                            'relative_height_m',
                            'horizontal_distance_m',
                        ]
                        
                        with open(csv_path + '.csv', 'w+', encoding='UTF8', newline='') as csvf:
                            csv_writer = writer(csvf)
                            csv_writer.writerow(header)

                    for key in det_dict:
                        for det in det_dict[key]:
                            # write detection entries
                            with open(csv_path + '.csv', 'a', encoding='UTF8', newline='') as csvf:
                                csv_writer = writer(csvf)
                                csv_writer.writerow(det)

                # Evaluate warning signs for this frame
                im0 = display_warning(im0,warning, mute_warnings)
                im0 = display_zoom_warning(im0, zoom_warning)

            else: # if no detections add 0 to all lists
                crane_height_queue.enqueue(0)
                helmet_height_queue.enqueue(0)
                helmet_detect_queue.enqueue({})

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--skip-step', type=int, default=1, help='default read every frame for streams')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--height-diff-thres', type=float, default=4.2, help='height difference threshold (meters)')
    parser.add_argument('--crane-width-queue-args', nargs='+', type=int, default=[10,2], help='crane queue arguments')
    parser.add_argument('--helmet-width-queue-args', nargs='+', type=int, default=[10,2], help='helmet queue arguments')
    parser.add_argument('--helmet-detect-queue-args', nargs='+', type=int, default=[10,1], help='helmet detect queue arguments')
    parser.add_argument('--focal-length-mm', type=float, default=4.0, help='camera focal length in millimeters')
    parser.add_argument('--pixel-size-mm', type=float, default=0.00112, help='sensor pixel size in millimeters per pixel')
    parser.add_argument('--helmet-height-mm', type=float, default=240.0, help='physical helmet height in millimeters')
    parser.add_argument('--hook-height-mm', type=float, default=300.0, help='physical hook height in millimeters')
    parser.add_argument('--camera-height-m', type=float, default=10.0, help='installation height of the camera in meters')
    parser.add_argument('--min-zoom-level', type=int, default=10, help='minimum camera zoom level')
    parser.add_argument('--max-zoom-level', type=int, default=25, help='maximum camera zoom level')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    parser.add_argument('--mute-warnings', action='store_true', help='mute warning sound')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1      # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
