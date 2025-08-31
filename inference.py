import torch
from torchvision import transforms
from PIL import Image
import json
import os
import pandas as pd
from tqdm import tqdm

from utils.utils import device_settings

# 디바이스 설정
device, _, _, _ = device_settings()

# ✅ 모델 로드
model = torch.load("convnext.pt", weights_only=False)
model = model.to(device)
model.eval()

with open('./data/label/AICamp-2023-Skin-Conditions_Dataset_int2label.json', 'r') as f:
    int_to_label = json.load(f)['int_to_label']


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ConvNeXt의 기본 입력 사이즈
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


inference_dirs = "./data/crawling_data/test"

res_imgs = []
for imgs in tqdm(os.listdir(inference_dirs), desc="Inference", unit="image"):
    image_path = os.path.join(inference_dirs, imgs)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = int_to_label[str(pred_idx)]
        pred_prob = probs[0][pred_idx].item()

    # print(f"🧠 예측 클래스: {pred_class}")
    # print(f"🔢 확률: {pred_prob:.4f}")

    res_imgs.append({
        "image": imgs,
        "predicted_label": pred_class,
        "probability": pred_prob
    })

pd.DataFrame(res_imgs).to_csv("inference_result.csv", index=False)

