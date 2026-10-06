# 🩺 Early-Stage Diabetes Prediction: A Comparative ML Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aframusarratdiya/diabetes-prediction-ml/blob/main/Diabetes_Prediction_ML_Project.ipynb)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

An empirical clinical classification benchmark comparing 9 predictive architectures across ~100,000 patient records. This study addresses class imbalance, data leakage risks, and model generalization to guide clinical decision support systems.

---

## 📌 Executive Summary

Early clinical triage for chronic conditions like Type 2 Diabetes prioritizes **high recall (sensitivity)** to reduce false negatives—an undetected diabetic patient faces high risks of irreversible organ complications, whereas a false positive merely leads to a confirmatory lab test.

### Core Project Highlights:
* **Real-World Imbalance Treatment:** Evaluated **inverse class-frequency weighting** against **SMOTE**. Synthetic oversampling caused severe memorization (train-test F1 gaps of 0.14–0.30), while class weighting brought the generalization gap below **0.02** for 7 of the 9 models.
* **Architecture Comparison:** Evaluated models ranging from regularized linear baselines to tree ensembles (Random Forest, XGBoost, Gradient Boosting, LightGBM), deep hybrid neural networks (DNet: 1D-CNN + LSTM), and a meta-stacking ensemble.
* **Clinical Trade-offs:** **LightGBM** reached the highest recall (**94.7%**) for patient screening, while **Gradient Boosting** achieved the best overall balanced performance (**F1: 0.853**, **Accuracy: 97.8%**, **ROC-AUC: 0.988**).
* **Critical Data Leakage Analysis:** Identified diagnostic threshold leakage driven by HbA1c and fasting blood glucose, demonstrating the need to separate diagnostic criteria from pre-diagnostic screening variables.

---

## 📊 Dataset & Preprocessing Pipeline

* **Data Source:** [100,000 Diabetes Clinical Dataset](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset) (US Clinical Health Survey, 2015–2022).
* **Target Distribution:** 91.4% non-diabetic ($n=90,571$) vs. 8.6% diabetic ($n=8,500$) (~10:1 imbalance ratio).
* **Validation Strategy:** Stratified 80/20 train/test split (79,256 train / 19,815 test) maintaining class distribution across partitions.

1. **Outlier-Resilient Scaling:** Applied `RobustScaler` (median & interquartile range) across continuous features (`age`, `bmi`, `hbA1c_level`, `blood_glucose_level`) to prevent leverage points from skewing linear and distance-based estimators.
2. **Domain-Specific Encoding:** Encoded smoking history into an ordinal clinical scale (`never`: 0, `no info`: 1, `former/ever`: 2, `current`: 3) and removed low-representation categories.
3. **Dimensionality Reduction:** Dropped collection year, geographic location, and demographic race tags after correlation analyses showed negligible signal ($r < 0.01$) relative to the target label.

---

## 🔬 Benchmark Results

Evaluated on the unseen test set (19,815 records):

| Model Family | Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Generalization Gap (Train-Test F1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Boosting** | **Gradient Boosting** | **0.978** | **0.980** | 0.755 | **0.853** | **0.988** | **-0.001 (Optimal)** |
| **Boosting** | **LightGBM** | 0.923 | 0.528 | **0.947** | 0.678 | **0.988** | **-0.001 (Optimal)** |
| **Ensemble** | **Stacking Classifier** | 0.933 | 0.566 | 0.934 | 0.705 | **0.988** | **-0.009 (Optimal)** |
| **Boosting** | XGBoost | 0.926 | 0.539 | 0.942 | 0.686 | **0.988** | +0.139 (Overfitting) |
| **Deep Learning** | DNet (CNN-LSTM) | 0.923 | 0.528 | 0.914 | 0.670 | 0.981 | **-0.009 (Optimal)** |
| **Bagging** | Random Forest | 0.949 | 0.648 | 0.878 | 0.746 | 0.984 | +0.068 (Slight) |
| **Neural Net** | MLP (64-32) | 0.971 | 0.977 | 0.679 | 0.801 | 0.983 | +0.012 (Optimal) |
| **Kernel** | SVM (RBF Kernel) | 0.908 | 0.482 | 0.930 | 0.635 | 0.975 | **-0.003 (Optimal)** |
| **Baseline** | Logistic Regression | 0.897 | 0.449 | 0.891 | 0.597 | 0.966 | **-0.010 (Optimal)** |

---

## 💡 Practical Insights

### 1. Model Selection Depends on Deployment Context
* **First-Stage Screening Triage:** **LightGBM** is the recommended model. Its **94.7% recall** minimizes missed diabetic cases, matching clinical triage goals where false negatives are costly.
* **Automated Diagnosis / Treatment Planning:** **Gradient Boosting** is the better choice when minimizing unnecessary clinic visits, delivering **98.0% precision** alongside an **0.853 F1-score**.

### 2. The Synthetic Sampling Pitfall (SMOTE vs. Class Weights)
* Generating synthetic minority samples with SMOTE created artificial density neighborhoods that models easily memorized, causing validation drops of up to 30%.
* Using **inverse frequency class weighting** adjusted the loss penalty directly on actual patient data, resolving the generalization gap without data distortion.

### 3. Feature Importance & Clinical Leakage
* Feature correlation aligned with diagnostic physiology: `blood_glucose_level` ($r=0.42$) and `hbA1c_level` ($r=0.40$) dominated tree split criteria, followed by `age` ($r=0.26$) and `bmi` ($r=0.21$).
* Because clinical standards define diabetes using HbA1c $\ge 6.5\%$ and fasting glucose $\ge 126\text{ mg/dL}$, models relying on these variables can show target leakage. A follow-up iteration should evaluate pre-screening solely on non-invasive metrics (lifestyle, demographics, BMI, family history).

---

## 🛠️ Tech Stack & Architecture

* **Environment:** Python 3.10+, Jupyter / Google Colab
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
* **Deep Learning:** `tensorflow` / `keras` (1D-CNN + stacked LSTM architecture)
* **Data & Analytics:** `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`

---

## 📖 Research Context & Citation

📄 **Read the full research paper:** [Diabetes_Prediction_Using_MachineLearning.pdf](./Diabetes_Prediction_Using_MachineLearning.pdf)

> **"Diabetes Prediction Using Machine Learning: A Comparative Study of Classification Algorithms"**[cite: 8]  
> *Afra Musarrat Diya* — Department of Computer Science, Brac University[cite: 8].
