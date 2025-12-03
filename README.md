# 🌿 Plant Disease Classification using Hybrid AI

This project implements a hybrid machine learning approach to classify plant diseases from leaf images. It combines Deep Learning (ResNet18 or ResNet50) for feature extraction with a classical Machine Learning model (Random Forest or Gradient Boosting) for efficient classification.

## 🚀 Features

*   **Hybrid Architecture:** Uses CNNs (ResNet) for features and ML classifiers (RF/GB) for decision making.
*   **GPU Acceleration:** Automatically detects and uses CUDA (NVIDIA GPU) for feature extraction if available.
*   **Flexible Modes:** Switch between "Basic" (Fast) and "Enhanced" (Accurate) modes easily.
*   **Custom Configuration:** Mix and match models (ResNet18/50) and classifiers (RF/GB).
*   **Performance Logging:** Automatically logs experiments, timing, and accuracy to `src/experiment_log.txt`.

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
│   ├── classify.py
│   ├── notebook.ipynb
│   └── experiment_log.txt
├── dataset/
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Potato___healthy/
│   └── ... (other class folders)
├── requirements.txt
└── README.md
```

## 📦 Installation & Requirements

It is highly recommended to use a Virtual Environment.

1.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    > **💡 GPU Support:** If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch.
    > ```bash
    > pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    > ```

## 🏃 Usage

You can run the script with various configurations.

### 1. Basic Mode (Default)
Uses **ResNet18** + **Random Forest**. Limits data to **200 images/class** for speed.
```bash
python src/classify.py
# OR
python src/classify.py basic
```

### 2. Enhanced Mode
Uses **ResNet50** + **Gradient Boosting**. Uses **ALL** images for maximum accuracy.
```bash
python src/classify.py enhanced
```

### 3. Custom Configuration
You can mix and match models and classifiers.
Format: `python src/classify.py [MODEL] [CLASSIFIER] [LIMIT]`

*   **Models:** `18` (ResNet18), `50` (ResNet50)
*   **Classifiers:** `RF` (Random Forest), `GB` (Gradient Boosting)
*   **Limit:** Number of images per class (`0` for ALL).

**Examples:**
```bash
# ResNet18 + Gradient Boosting + All Data
python src/classify.py 18 GB 0

# ResNet50 + Random Forest + 500 images per class
python src/classify.py 50 RF 500
```

## 📊 Experiment Logs
Every time you run the script, the results are saved to `src/experiment_log.txt`. This file contains:
*   Timestamp & Configuration used.
*   Execution time for each step.
*   Overall Accuracy.
*   Detailed Classification Report (Precision, Recall, F1-Score).

### 📓 Interactive Notebook
If you prefer an interactive environment, use the provided Jupyter Notebook. It has been updated to support the new configurations.

```bash
jupyter notebook src/notebook.ipynb
```
In the notebook, you can set `MODEL_TYPE`, `CLASSIFIER_TYPE`, and `DATA_LIMIT` variables at the top to customize your run.

## 📉 Understanding the Confusion Matrix

To evaluate how well the model performs, we use a **Confusion Matrix**. This chart visualizes exactly where the model is "confused" between different diseases.

### How to Read It
* **Y-Axis (Left):** The **True Label** (What the plant actually has).
* **X-Axis (Bottom):** The **Predicted Label** (What the AI thought it was).

### 1. The "Perfect" Diagonal
Ideally, we want to see high numbers (dark blue squares) running in a diagonal line from the top-left to the bottom-right.

### 2. Where the Model Get Confused
Numbers **outside** that main diagonal represent errors. By looking at these, we can see *how* the model messed up.
* **Example:** *Early Blight*, *Late Blight*, and *Bacterial Spot* often look very similar (brown spots), making it harder for the AI to distinguish them compared to distinct viral infections.

