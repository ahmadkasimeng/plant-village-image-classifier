# Plant Disease Classification using Hybrid AI

This project implements a hybrid machine learning approach to classify plant diseases from leaf images. It combines Deep Learning (ResNet18 or ResNet50) for feature extraction with a classical Machine Learning model (Random Forest) for efficient classification.

## 🚀 How It Works

1.  **Feature Extraction:** We use a pre-trained **ResNet18** neural network. There is also a script for **ResNet50**. We remove the final classification layer, turning the network into a powerful "feature extractor" that converts image data into numerical vectors.
2.  **Classification:** These vectors are fed into a **Random Forest Classifier** to determine the specific disease.

## 📂 Dataset Setup (Important)

This project requires the **PlantVillage** dataset. Because the dataset is large, it is not included in this repository.

1.  **Download:** Download the dataset from Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease/data)
2.  **Extract & Organize:**
      * Do **not** place the raw `PlantVillage` folder directly into `dataset/`.
      * Open the downloaded folder, find the **inner** folders (the ones named like `Potato___healthy`, `Pepper__bell___Bacterial_spot`), and move those specific folders into the `dataset/` directory.

**Correct Structure:**

```text
Project_Root/
├── src/
│   └── classify.py
├── dataset/
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Potato___healthy/
│   └── ... (other class folders)
├── requirements.txt
└── README.md
```

## 📦 Installation & Requirements

It is highly recommended to use a Virtual Environment to avoid conflicts with your global Python installation. Python313 was used.

1.  **Create a Virtual Environment (Optional but Recommended):**

    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```
    

2.  **Install Dependencies:**
    This project uses a `requirements.txt` file generated for this specific script.

    ```bash
    pip install -r requirements.txt
    ```

    > **💡 PyTorch Note:** The requirements file includes `torch` and `torchvision`. If you do not have a dedicated GPU, or if you want to save space and install the lighter **CPU-only** version, run this command *after* installing requirements:

    > ```bash
    > pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    > ```

## ⚙️ Customizing the Script

You can tweak parameters in `src/classify.py` to balance speed vs. accuracy:

  * **Sample Size:**
      * Currently, the script limits processing to **200 images per class** for speed (`[:200]`).
      * To use the full dataset, remove the slice `[:200]` in the `load_data_with_ai` function.
  * **Model Complexity:**
      * You can adjust `n_estimators` (number of trees) in the `RandomForestClassifier`. Higher numbers (e.g., 500 or 1000) are more stable but slower.
  * **Feature Extractor:**
      * The script uses `resnet18`. You can swap this for `resnet50` if you need deeper feature analysis, though it will be slower on CPU.

## 🏃 Usage

Run the classification script from the root or src folder:

```bash
python src/classify.py
```

### 📓 Interactive Notebook

If you prefer an interactive environment to visualize the steps or tweak the model without reloading the dataset every time, a Jupyter Notebook is provided.

1.  Ensure you have Jupyter installed (`pip install notebook`).
2.  Navigate to the `src` folder.
3.  Open the notebook:
    ```bash
    jupyter notebook notebook.ipynb
    ```

The script will:

1.  Load and process images.
2.  Train the Random Forest model.
3.  Output accuracy metrics and a classification report.
```bash
 🧠 Loading ResNet18 (Pre-trained AI)...
⏳ Processing images with AI features...
   Extracting features for: Pepper__bell___Bacterial_spot
   Extracting features for: Pepper__bell___healthy
   Extracting features for: Potato___Early_blight
   Extracting features for: Potato___healthy
   Extracting features for: Potato___Late_blight
   Extracting features for: Tomato_Bacterial_spot
   Extracting features for: Tomato_Early_blight
   Extracting features for: Tomato_healthy
   Extracting features for: Tomato_Late_blight
   Extracting features for: Tomato_Leaf_Mold
   Extracting features for: Tomato_Septoria_leaf_spot
   Extracting features for: Tomato_Spider_mites_Two_spotted_spider_mite
   Extracting features for: Tomato__Target_Spot
   Extracting features for: Tomato__Tomato_mosaic_virus
   Extracting features for: Tomato__Tomato_YellowLeaf__Curl_Virus

✅ Data Processed. Shape: (2952, 512)
🌲 Training Optimized Random Forest (500 Trees)...
📊 Evaluating...
✨ Accuracy: 82.23%

Classification Report:
                                             precision    recall  f1-score   support

              Pepper__bell___Bacterial_spot       0.91      0.89      0.90        45
                     Pepper__bell___healthy       0.85      0.97      0.91        40
                      Potato___Early_blight       0.90      0.96      0.93        54
                       Potato___Late_blight       0.87      0.79      0.83        43
                           Potato___healthy       0.86      0.90      0.88        20
                      Tomato_Bacterial_spot       0.68      0.86      0.76        37
                        Tomato_Early_blight       0.55      0.44      0.49        41
                         Tomato_Late_blight       0.90      0.51      0.65        35
                           Tomato_Leaf_Mold       0.80      0.92      0.85        38
                  Tomato_Septoria_leaf_spot       0.61      0.70      0.65        33
Tomato_Spider_mites_Two_spotted_spider_mite       0.83      0.94      0.88        36
                        Tomato__Target_Spot       0.84      0.71      0.77        51
      Tomato__Tomato_YellowLeaf__Curl_Virus       0.91      0.91      0.91        32
                Tomato__Tomato_mosaic_virus       0.92      0.87      0.89        39
                             Tomato_healthy       0.92      0.94      0.93        47

                                   accuracy                           0.82       591
                                  macro avg       0.82      0.82      0.81       591
                               weighted avg       0.83      0.82      0.82       591

🎨 Plotting Confusion Matrix...
 ```
4.  Display a Confusion Matrix heatmap.


![Resulting Confusion Matrix](Confusion%20Matrix.png "Optional title")



---

# Confusion Matrix for all dataset (Not constrained to 200)

![Resulting Confusion Matrix](Confusion%20Matrix%20All%20Dataset.png "Optional title")

---

## 📉 Understanding the Confusion Matrix

To evaluate how well the model performs, we use a **Confusion Matrix**. This chart visualizes exactly where the model is "confused" between different diseases. Using the unconstrained version as a guide.

### How to Read It
* **Y-Axis (Left):** The **True Label** (What the plant actually has).
* **X-Axis (Bottom):** The **Predicted Label** (What the AI thought it was).

### 1. The "Perfect" Diagonal
Ideally, we want to see high numbers (dark blue squares) running in a diagonal line from the top-left to the bottom-right.
* **Example:** Look at **`Tomato_Tomato_YellowLeaf__Curl_Virus`** (second from bottom). The model correctly predicted this **624 times**. The rest of that row is mostly zeros, meaning the model is very confident and accurate for this disease.

### 2. Where the Model Get Confused
Numbers **outside** that main diagonal represent errors. By looking at these, we can see *how* the model messed up.

**Real Example from our Results:**
Look at the row for **`Tomato_Early_blight`**. This is a tricky class for the AI:
* It correctly identified it **60** times.
* It incorrectly guessed **`Tomato_Bacterial_spot`** **38** times.
* It incorrectly guessed **`Tomato_Late_blight`** **40** times.

**Why?** Visually, *Early Blight*, *Late Blight*, and *Bacterial Spot* all look very similar (brown spots on leaves), making it harder for the AI to distinguish them compared to distinct viral infections.
