import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
script_dir = Path(__file__).resolve().parent
dataset_path = script_dir.parent / 'dataset'

# --- STEP 1: SETUP THE AI FEATURE EXTRACTOR ---
print("🧠 Loading ResNet18 (Pre-trained AI)...")
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) # Remove last layer
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(img_path):
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = resnet(input_batch)
    return output.squeeze().numpy()

def load_data_with_ai(base_path):
    features_list = []
    labels_list = []
    class_names = os.listdir(base_path)
    
    print("⏳ Processing images with AI features...")
    
    for class_name in class_names:
        if class_name == "PlantVillage": continue
        
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir): continue
        
        file_list = os.listdir(class_dir) # Limit for speed
        print(f"   Extracting features for: {class_name}")
        
        for img_name in file_list:
            img_path = os.path.join(class_dir, img_name)
            try:
                vector = get_vector(img_path)
                features_list.append(vector)
                labels_list.append(class_name)
            except Exception as e:
                print(f"Error reading {img_name}: {e}")
                
    return np.array(features_list), np.array(labels_list)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Extract Features
    X, y = load_data_with_ai(dataset_path)
    print(f"\n✅ Data Processed. Shape: {X.shape}")

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Optimized Random Forest
    # Increased trees to 500 and added class_weight='balanced'
    print("🌲 Training Optimized Random Forest (500 Trees)...")
    rf_model = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    # 4. Evaluation
    print("📊 Evaluating...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✨ Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Visualization (Confusion Matrix)
    print("🎨 Plotting Confusion Matrix...")
    unique_classes = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()