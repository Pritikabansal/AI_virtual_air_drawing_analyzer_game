# predict.py (updated for import)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return features.squeeze()

def load_feature_db(feature_db_path):
    with open(feature_db_path, "rb") as f:
        return pickle.load(f)

def predict(user_img_path, feature_db, classes_to_check, threshold=0.60):
    user_feat = extract_feature(user_img_path).to(device)

    similarities = {}
    for path, feat in feature_db.items():
        feat = feat.to(device)
        sim = F.cosine_similarity(user_feat, feat, dim=0).item()
        similarities[path] = sim

    # Filter similarities based on classes to check
    filtered_similarities = {path: sim for path, sim in similarities.items() 
                              if os.path.basename(os.path.dirname(path)) in classes_to_check}

    if not filtered_similarities:
        return "unknown", 0.0  # No valid classes found

    best_match_path = max(filtered_similarities, key=filtered_similarities.get)
    best_score = filtered_similarities[best_match_path]
    best_class = os.path.basename(os.path.dirname(best_match_path))

    if best_score < threshold:
        return "unknown", best_score

    return best_class, best_score


# Only run as script for debug/testing:
if __name__ == "__main__":
    user_image_path = "D:/AI_virtual_air/realistic/heart.png"
    features_pkl = "D:/AI_virtual_air/features.pkl"

    feature_db = load_feature_db(features_pkl)

    predicted_class, confidence = predict(user_image_path, feature_db)

    if predicted_class == "unknown":
        print("\n \U0001F972 Your image does not look like a meaningful drawing. Try again!")
    else:
        print(f"\n✅ Prediction: {predicted_class} ({confidence * 100:.0f}% Accurate)")
