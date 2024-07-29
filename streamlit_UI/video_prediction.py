import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F
import argparse
from PIL import ImageFont, ImageDraw
from ultralytics import YOLO  
import cv2
from preprocessing.tool.YoloV5_Crop_Frame.models.experimental import attempt_load
from preprocessing.tool.YoloV5_Crop_Frame.utils.torch_utils import select_device
from preprocessing.tool.YoloV5_Crop_Frame.utils.general import non_max_suppression
from model.xmodel import XMT


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)
# Define the XMT model
model = XMT(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024)
model.to(device)

checkpoint = torch.load('/Users/lap01743/Downloads/Weightxmodel_Tranformers_Deepfakes_DectionVideoImage/xmodel_deepfake_weight.pth', map_location=torch.device('cpu'))
filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)
model.eval()
print("load model successfully")



# Image preprocessing transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def draw_box_and_label(image, box, label):

    draw = ImageDraw.Draw(image)
    box = [int(coordinate) for coordinate in box]
    expanded_box = [box[0], box[1] - 20, box[2], box[3] + 20]
    box_tuple = (expanded_box[0], expanded_box[1], expanded_box[2], expanded_box[3])
    color = "red" if label.startswith("Fake") else ("yellow" if label.startswith("Calculating") else "green")
    font = ImageFont.load_default(30)
    draw.rectangle(box_tuple, outline=color, width=2)
    text_position = (box[0], box[1] - 10) if box[1] - 10 > 0 else (box[0], box[1])
    draw.text(text_position, label, fill=color, font=font)
    return image

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()




def process_frame(frame):
    image = Image.fromarray(frame)    
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        for box in boxes:
            face = image.crop(box)
            face = np.array(face)
            face = normalize_transform(face).unsqueeze(0).to(device)

            prediction = model(face)
            prediction = torch.softmax(prediction, dim=1)
            pred_real_percentage = prediction[0][1].item() * 100
            pred_fake_percentage = prediction[0][0].item() * 100

            if max(pred_real_percentage, pred_fake_percentage) > 80:
                _, predicted_class = torch.max(prediction, 1)
                pred_label = predicted_class.item()
                label = "Real" if pred_label == 1 else "Fake"
            else:
                label = "Calculating"

            if label == "Calculating": 
                label_with_probabilities = f"{label}"
            elif label == "Fake" : 
                label_with_probabilities = f"{label}:{(100 - pred_real_percentage):.2f}%"
            else: 
                label_with_probabilities = f"{label}:{pred_real_percentage:.2f}%"
            draw_box_and_label(image, box, label_with_probabilities)
    return np.array(image)














