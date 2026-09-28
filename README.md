# 🎗️ Breast Cancer Classification with Neural Networks (Breast Cancer Wisconsin)

A machine learning project that builds a **neural network** with **TensorFlow/Keras** to distinguish between **malignant** and **benign** breast tumors based on features extracted from digitized biopsy images.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Disclaimer](#-disclaimer)

---

## 🔎 Overview

The goal is to train a binary classification model that predicts tumor type from 30 numeric features. The project includes:

- Loading, cleaning, and exploring the data.
- Splitting the data and standardizing feature scales.
- Building a baseline model and evaluating it with multiple metrics.
- Building an improved model using Dropout and a lower learning rate, then comparing it with the baseline.
- Testing predictions on individual samples.

## 📊 Dataset

| Item | Details |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Samples** | 569 |
| **Features** | 30 numeric features |
| **Target** | `diagnosis`: Malignant (M) or Benign (B) |
| **Missing values** | None |

**Class distribution:**

- 🟢 Benign: **357** samples
- 🔴 Malignant: **212** samples

The 30 features are ten cell nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each computed in three forms: **mean**, **standard error (se)**, and **worst**.

> Place `data.csv` in the same directory as the notebook.

## 🛠️ Project Workflow

1. **Import libraries** and set random seeds (`seed = 42`) for reproducibility.
2. **Load the data** and drop the non-informative columns (`id` and `Unnamed: 32`).
3. **Encode the target:** Malignant `M → 0`, Benign `B → 1`.
4. **Explore:** check class distribution and confirm there are no missing values.
5. **Split:** 80% training (455 samples) and 20% testing (114 samples), using `stratify` to preserve class ratios.
6. **Scale:** `StandardScaler` is fit on the training data only to avoid data leakage.
7. **Build and train** the baseline model with `EarlyStopping`.
8. **Evaluate:** accuracy, classification report, confusion matrix, and ROC curve.
9. **Improve** the model and compare it with the baseline.
10. **Predict** on samples from the test set.

## 🧠 Model Architectures

### Baseline Model

| Layer | Size | Activation |
|---|---|---|
| Input | 30 | — |
| Dense | 16 | ReLU |
| Dense | 8 | ReLU |
| Dense (Output) | 1 | Sigmoid |

- **Parameters:** 641
- **Optimizer:** Adam (`lr = 0.001`)
- **Loss:** Binary Crossentropy
- **Training:** up to 50 epochs, `batch_size = 16`, `validation_split = 0.2`, with `EarlyStopping` (patience 10)

### Improved Model (v2)

| Layer | Size | Activation |
|---|---|---|
| Input | 30 | — |
| Dense | 32 | ReLU |
| Dropout | 0.3 | — |
| Dense | 16 | ReLU |
| Dropout | 0.2 | — |
| Dense (Output) | 1 | Sigmoid |

- **Optimizer:** Adam (`lr = 0.0005`)
- **Training:** up to 150 epochs, `batch_size = 32`, with `EarlyStopping` (patience 15)

## 🏆 Results

### Model Comparison (Test Set)

| Model | Accuracy |
|---|---|
| Baseline | **96.49%** |
| Improved (v2) | **97.37%** |

### Classification Report (Baseline)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Malignant | 0.93 | 0.98 | 0.95 | 42 |
| Benign | 0.99 | 0.96 | 0.97 | 72 |
| **Accuracy** | | | **0.96** | 114 |

- Test loss: **0.0932**
- The high recall for the malignant class (0.98) is important in medical applications, since it reduces the chance of missing a malignant case.

## 📦 Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow (2.x)

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## 🚀 How to Run

### Option 1: Google Colab (easiest)

1. Click the **Open In Colab** badge at the top.
2. Run the cells in order.
3. When the file upload prompt appears, upload `data.csv`.

### Option 2: Run locally

```bash
# 1) Clone the repository
git clone https://github.com/rahafalnjjar49-cloud/Breast-Cancer-Wisconsin.git
cd Breast-Cancer-Wisconsin

# 2) Install dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter

# 3) Place data.csv in the same folder, then launch the notebook
jupyter notebook Breast_Cancer_Wisconsin.ipynb
```

> ⚠️ The second cell in the notebook uses `google.colab.files.upload()`, which only works inside Colab. When running locally, replace it with:
> ```python
> df_raw = pd.read_csv('data.csv')
> ```

## 📁 Project Structure

```
Breast-Cancer-Wisconsin/
├── Breast_Cancer_Wisconsin.ipynb   # Main notebook
├── data.csv                        # Dataset
└── README.md                       # This file
```

## ⚠️ Disclaimer

This project is for **educational purposes only** and must not be used for real medical diagnosis. The model is not a substitute for professional medical advice.

---

⭐ If you found this project useful, consider giving the repository a star!
