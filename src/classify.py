import os
import time
from datetime import datetime
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
# Check for GPU availability
print("🔍 Checking for GPU availability...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}") 
if device.type == 'cuda':
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

print("🧠 Loading ResNet18 (Pre-trained AI)...")
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) # Remove last layer

# MOVE MODEL TO GPU
resnet = resnet.to(device) 
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
    
    # MOVE INPUT TO GPU
    input_batch = input_tensor.unsqueeze(0).to(device) 
    
    with torch.no_grad():
        output = resnet(input_batch)
    
    # MOVE OUTPUT BACK TO CPU FOR NUMPY
    return output.cpu().squeeze().numpy()

def load_data_with_ai(base_path):
    features_list = []
    labels_list = []
    class_names = os.listdir(base_path)
    
    print("⏳ Processing images with AI features...")
    
    for class_name in class_names:
        if class_name == "PlantVillage": continue
        
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir): continue
        
        file_list = os.listdir(class_dir)[:200] 
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
    # Setup Logging
    log_file_path = script_dir / "performance_log.txt"
    timings = {}
    
    # Get Device Name for Log
    device_name = "CPU"
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(0)

    print(f"\n⏱️  Starting Performance Tracking on {device_name}...")

    # 1. Extract Features (NOW ON GPU)
    start_time = time.time()
    X, y = load_data_with_ai(dataset_path)
    timings["Feature Extraction (ResNet18)"] = time.time() - start_time
    print(f"✅ Data Processed. Shape: {X.shape}")
    print(f"⏱️  Feature Extraction took: {timings['Feature Extraction (ResNet18)']:.2f}s")

    # 2. Split Data
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    timings["Data Splitting"] = time.time() - start_time

    # 3. Train Optimized Random Forest (STILL CPU - Scikit-learn limit)
    print("🌲 Training Optimized Random Forest (500 Trees)...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1) 
    # Added n_jobs=-1 to use all CPU cores
    rf_model.fit(X_train, y_train)
    timings["Classifier Training (RF)"] = time.time() - start_time
    print(f"⏱️  Training took: {timings['Classifier Training (RF)']:.2f}s")

    # 4. Evaluation
    print("📊 Evaluating...")
    start_time = time.time()
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    timings["Evaluation"] = time.time() - start_time
    
    print(f"✨ Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- LOGGING TO FILE ---
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_exists = log_file_path.exists()
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            if not log_exists:
                # Write Header
                f.write(f"{'Timestamp':<20} | {'Device':<25} | {'Task':<30} | {'Duration (s)':<15}\n")
                f.write("-" * 95 + "\n")
            
            for task, duration in timings.items():
                f.write(f"{timestamp:<20} | {device_name:<25} | {task:<30} | {duration:<15.4f}\n")
            f.write("-" * 95 + "\n")
        print(f"\n📝 Performance log updated: {log_file_path}")
    except Exception as e:
        print(f"⚠️ Failed to write log: {e}")

    # 5. Visualization
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