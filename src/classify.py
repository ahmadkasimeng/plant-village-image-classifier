import os
import time
import argparse
from datetime import datetime
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
script_dir = Path(__file__).resolve().parent
dataset_path = script_dir.parent / 'dataset'
log_file_path = script_dir / "experiment_log.txt"

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Plant Disease Classification')
parser.add_argument('config', nargs='*', default=['basic'], 
                    help='Configuration: "basic", "enhanced", "18 RF", "18 GB", "50 RF", "50 GB". Optional last arg: limit (0=all, default=200)')
args = parser.parse_args()

# Parse Configuration
config_input = [x.upper() for x in args.config]
model_type = "RESNET18"
clf_type = "RF"
data_limit = 200
mode_name = "Basic"

# Check for limit argument at the end
if len(config_input) > 0 and config_input[-1].isdigit():
    limit_arg = int(config_input.pop())
    data_limit = None if limit_arg == 0 else limit_arg
    print(f"🔢 Data Limit Set to: {'ALL' if data_limit is None else data_limit}")

if len(config_input) == 0 or config_input == ['BASIC']:
    mode_name = "Basic"
    model_type = "RESNET18"
    clf_type = "RF"
    # Default limit is already 200, unless overridden above
elif config_input == ['ENHANCED']:
    mode_name = "Enhanced"
    model_type = "RESNET50"
    clf_type = "GB"
    # Enhanced defaults to ALL (None) if not specified
    if 'limit_arg' not in locals():
        data_limit = None
elif len(config_input) == 2:
    # Custom Mode
    if config_input[0] == '18': model_type = "RESNET18"
    elif config_input[0] == '50': model_type = "RESNET50"
    
    if config_input[1] == 'RF': clf_type = "RF"
    elif config_input[1] == 'GB': clf_type = "GB"
    
    mode_name = f"Custom ({model_type} + {clf_type})"
    # Custom defaults to ALL (None) if not specified
    if 'limit_arg' not in locals():
        data_limit = None
else:
    print(f"⚠️ Unrecognized configuration: {args.config}. Defaulting to Basic.")

# --- STEP 1: SETUP THE AI FEATURE EXTRACTOR ---
# Check for GPU availability
print("🔍 Checking for GPU availability...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}") 
if device.type == 'cuda':
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

print(f"🧠 Loading {model_type}...")
if model_type == 'RESNET50':
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
else:
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

def load_data_with_ai(base_path, limit=200):
    features_list = []
    labels_list = []
    class_names = os.listdir(base_path)
    
    print(f"⏳ Processing images with {model_type} features...")
    
    for class_name in class_names:
        if class_name == "PlantVillage": continue
        
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir): continue
        
        all_files = os.listdir(class_dir)
        # Apply limit if provided (None means all files)
        file_list = all_files[:limit] if limit else all_files
        
        print(f"   Extracting features for: {class_name} ({len(file_list)} images)")
        
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
    timings = {}
    
    # Get Device Name for Log
    device_name = "CPU"
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(0)

    print(f"\n⏱️  Starting Performance Tracking on {device_name}...")
    print(f"⚙️  Mode: {mode_name}")

    # 1. Extract Features (NOW ON GPU)
    start_time = time.time()
    
    X, y = load_data_with_ai(dataset_path, limit=data_limit)
    
    timings[f"Feature Extraction ({model_type})"] = time.time() - start_time
    
    if len(X) == 0:
        print("❌ No data loaded. Exiting.")
        exit()
        
    print(f"✅ Data Processed. Shape: {X.shape}")
    print(f"⏱️  Feature Extraction took: {timings[f'Feature Extraction ({model_type})']:.2f}s")

    # 2. Split Data
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    timings["Data Splitting"] = time.time() - start_time

    # 3. Train Classifier
    start_time = time.time()
    
    if clf_type == 'GB':
        print("🚀 Training Gradient Boosting Model...")
        model = HistGradientBoostingClassifier(
            max_iter=1000,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42
        )
    else:
        print("🌲 Training Optimized Random Forest...")
        model = RandomForestClassifier(
            n_estimators=500, 
            class_weight='balanced', 
            random_state=42, 
            n_jobs=-1
        )
        
    model.fit(X_train, y_train)
    timings[f"Classifier Training ({clf_type})"] = time.time() - start_time
    print(f"⏱️  Training took: {timings[f'Classifier Training ({clf_type})']:.2f}s")

    # 4. Evaluation
    print("📊 Evaluating...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    timings["Evaluation"] = time.time() - start_time
    
    print(f"✨ Accuracy: {acc * 100:.2f}%")
    report_str = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report_str)

    # --- LOGGING TO FILE ---
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Configuration: {mode_name}\n")
            f.write(f"Dataset Limit: {'ALL' if data_limit is None else data_limit}\n")
            f.write(f"Device: {device_name}\n")
            f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
            f.write("-" * 80 + "\n")
            f.write("Timing Breakdown:\n")
            for task, duration in timings.items():
                f.write(f"  - {task:<35}: {duration:.4f}s\n")
            f.write("-" * 80 + "\n")
            f.write("Detailed Classification Report:\n")
            f.write(report_str)
            f.write("\n" + "=" * 80 + "\n\n")
            
        print(f"\n📝 Experiment log updated: {log_file_path}")
    except Exception as e:
        print(f"⚠️ Failed to write log: {e}")

    # 5. Visualization
    print("🎨 Plotting Confusion Matrix...")
    unique_classes = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f'Confusion Matrix ({mode_name})\nAccuracy: {acc*100:.1f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()