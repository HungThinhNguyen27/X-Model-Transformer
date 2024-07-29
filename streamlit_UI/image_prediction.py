import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import torch.nn.functional as F
import sys
sys.path.append('/Users/lap01743/Downloads/WorkSpace/capstone_wed/XMT_Model/streamlit_UI')
# from streamlit_UI.load_model import load_model_xmt
from PIL import ImageFont, ImageDraw
from model.xmodel import XMT


device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)
# Define the XMT model
model = XMT(image_size=224, patch_size=7, num_classes=2, channels=1024, dim=1024, depth=6, heads=8, mlp_dim=2048, gru_hidden_size=1024)
model.to(device)
# Load the pre-trained weights for the XMT model
checkpoint = torch.load('/Users/lap01743/Downloads/Weightxmodel_Transformers_Deepfakes_DectionVideoImage/xmodel_deepfake_weight.pth', map_location=torch.device('cpu'))
filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)

# Put the model in evaluation mode
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

def process_and_save_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    predictions_list = []

    # Detect faces
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
            # print("pred_real_percentage", pred_real_percentage)
            # print("pred_fake_percentage", pred_fake_percentage)
            _, predicted_class = torch.max(prediction, 1)
            pred_label = predicted_class.item()
            if pred_fake_percentage > 90: 
                label = "Fake"
            else: 
                label = "Real"
            draw_box_and_label(image, box, label)
            predictions_list.append(pred_label)
    return image

def draw_box_and_label(image, box, label):
    draw = ImageDraw.Draw(image)
    box = [int(coordinate) for coordinate in box]
    expanded_box = [box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10]
    box_tuple = (expanded_box[0], expanded_box[1], expanded_box[2], expanded_box[3])
    color = "red" if label.startswith("Fake") else "green"
    font = ImageFont.load_default(30)
    draw.rectangle(box_tuple, outline=color, width=2)
    text_position = (box[0], box[1] - 10) if box[1] - 10 > 0 else (box[0], box[1])
    draw.text(text_position, label, fill=color, font=font)

