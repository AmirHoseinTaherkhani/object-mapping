from ultralytics import YOLO
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO('D:\\Amir Taherkhani\\CosMos\\car_person\\Trial3\\runs\\detect\\train3\weights\\best.onnx')
results__ = model.track(source = 'Videos\\V5.mp4', show = False, tracker = 'bytetrack.yaml', save = True, save_dir = 'v5_tracked')