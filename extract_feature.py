# save_features_resnet50.py
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use ResNet50
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img)
    return features.squeeze().cpu()  # Save on CPU

def build_feature_db(dataset_dir):
    feature_dict = {}
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                feature = extract_feature(img_path)
                feature_dict[img_path] = feature
    return feature_dict

if __name__ == "__main__":
    dataset_path = "D:/AI_virtual_air/cartoon_dataset"  # Change if needed
    save_path = "D:/AI_virtual_air/features.pkl"

    print("🔄 Extracting features using ResNet50...")
    feature_db = build_feature_db(dataset_path)

    with open(save_path, "wb") as f:
        pickle.dump(feature_db, f)

    print("✅ Features saved to:", save_path)
