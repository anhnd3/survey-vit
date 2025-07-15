import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import Tuple
from tqdm import tqdm

# Number of emotion classes and their labels.
NUM_CLASSES = 7
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

def load_resnet50_model(model_path: str, device: torch.device) -> nn.Module:
    print(f"Loading ResNet50 model from {model_path} ...")
    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, NUM_CLASSES)
    resnet50 = resnet50.to(device)
    state_dict = torch.load(model_path, map_location=device)
    resnet50.load_state_dict(state_dict)
    resnet50.eval()
    print("ResNet50 model loaded successfully!")
    return resnet50

class ResNetEmotionModel:
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.classes = EMOTION_CLASSES

    def predict(self, image: Image.Image) -> dict:
        x = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        return {emotion: float(round(probabilities[i] * 100, 3)) for i, emotion in enumerate(self.classes)}