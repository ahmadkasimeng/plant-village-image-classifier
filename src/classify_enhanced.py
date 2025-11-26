import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
script_dir = Path(__file__).resolve().parent
dataset_path = script_dir.parent / 'dataset'

# --- STEP 1: SETUP THE AI FEATURE EXTRACTOR ---
# UPGRADE 2: Switching to ResNet50 (Deeper, more accurate features)
print("🧠 Loading ResNet50 (More powerful Pre-trained AI)...")
weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) # Remove last layer
resnet.eval()

# Move to GPU if available for faster feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on: {device}")
resnet.to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(img_path):
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device) # Move image to GPU/CPU
    
    with torch.no_grad():
        output = resnet(input_batch)
    
    # Move result back to CPU for Scikit-Learn
    return output.squeeze().cpu().numpy()

def load_data_with_ai(base_path):
    features_list = []
    labels_list = []
    
    # Check if dataset exists
    if not os.path.exists(base_path):
        print(f"❌ Error: Dataset not found at {base_path}")
        return np.array([]), np.array([])

    class_names = sorted(os.listdir(base_path))
    
    print("⏳ Processing images with ResNet50 features...")
    
    for class_name in class_names:
        if class_name.startswith('.'): continue # Skip hidden files
        
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir): continue
        
        # UPGRADE 3: Removed the [:200] slice to use ALL data
        file_list = os.listdir(class_dir) 
        print(f"   Extracting features for: {class_name} ({len(file_list)} images)")
        
        for img_name in file_list:
            img_path = os.path.join(class_dir, img_name)
            try:
                vector = get_vector(img_path)
                features_list.append(vector)
                labels_list.append(class_name)
            except Exception as e:
                # Silently skip bad images to keep output clean
                pass
                
    return np.array(features_list), np.array(labels_list)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Extract Features
    X, y = load_data_with_ai(dataset_path)
    
    if len(X) == 0:
        print("❌ No data loaded. Check your dataset folder structure.")
    else:
        print(f"\n✅ Data Processed. Shape: {X.shape}")

        # 2. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train Gradient Boosting Classifier
        # HistGradientBoostingClassifier is generally faster and more accurate than Random Forest
        # for this type of dense feature data.
        print("🚀 Training Gradient Boosting Model (This might take a moment)...")
        gb_model = HistGradientBoostingClassifier(
            max_iter=1000,          # More iterations usually equals better accuracy
            learning_rate=0.1,      # Standard learning rate
            class_weight='balanced',# Handle imbalanced classes
            random_state=42
        )
        gb_model.fit(X_train, y_train)

        # 4. Evaluation
        print("📊 Evaluating...")
        y_pred = gb_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"✨ Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 5. Visualization (Confusion Matrix)
        print("🎨 Plotting Confusion Matrix...")
        unique_classes = sorted(list(set(y)))
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=unique_classes, yticklabels=unique_classes)
        plt.title(f'Enhanced Confusion Matrix (ResNet50 + GradBoost)\nAccuracy: {acc*100:.1f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("confusion_matrix_results.png") 
        print("💾 Plot saved as confusion_matrix_results.png")
        plt.show()  