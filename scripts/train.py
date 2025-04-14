# %%
import matplotlib.pyplot as plt
import cv2
import os
from ultralytics import YOLO

# %%
# Load the model
model = YOLO("yolov8s.pt")

# Use the model
results = model.train(data=os.path.join('D:/Amir Taherkhani/CosMos/car_person/Trial3', 'data.yaml'), epochs = 100)

# %%
metrics = model.val(data=os.path.join('D:/Amir Taherkhani/CosMos/car_person/Trial3', 'data.yaml'))

# %%
testModel = model.val(data=os.path.join('D:/Amir Taherkhani/CosMos/car_person/Trial3', 'data.yaml'))

# %%
path = model.export(format="onnx")  # export the model to ONNX format


