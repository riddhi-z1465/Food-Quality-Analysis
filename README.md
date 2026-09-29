# Milk Quality Analysis and Classification Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, scientifically validated Machine Learning case-study project developed for **B.Tech Computer Science & Engineering (Semester V)**.

The system analyzes measurable physical, chemical, and sensory properties of milk (pH, temperature, taste, odor, fat, turbidity, colour) to classify quality as **Low**, **Medium**, or **High**, and deploys an interactive, production-ready **Streamlit web application**.

---

## 1. Problem Statement

Milk is an indispensable dietary component whose quality directly affects human health, consumer safety, and dairy manufacturing viability. The rapid onset of spoilage—triggered by bacterial proliferation, thermal abuse, acidification, and improper handling—demands continuous monitoring.

Traditional quality assessment methods, such as agar plate culturing, methylene blue reduction tests, and titration assays, are time-consuming (taking hours or days) and require laboratory facilities. This project establishes an automated, real-time classification system using seven measurable sensor indicators to deliver rapid, accurate quality grades.

---

## 2. Objectives

1. **End-to-End ML Pipeline:** Implement the complete lifecycle: Data Ingestion → Cleaning → Duplicate Audit → EDA → Preprocessing → Model Training → Cross-Validation → Benchmarking → Diagnostics → Streamlit Deployment.
2. **Scientific Data Integrity:** Identify and resolve the **Data Leakage Paradox** (976 duplicate rows in 1,059 raw records) to ensure honest, generalizable evaluations rather than memorized ~99.5% accuracy.
3. **Multi-Model Comparison:** Train, optimize, and cross-validate 5 distinct classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Random Forest Classifier
4. **Comprehensive Diagnostics:** Generate confusion matrices, model comparison charts, feature importance rankings, and root-cause error analysis on misclassifications.
5. **Interactive Deployment:** Deliver a professional Streamlit web application providing real-time quality inference, confidence distributions, EDA dashboards, and viva documentation.

---

## 3. Dataset Overview

* **Dataset Name:** Milk Quality Dataset
* **Source:** Open dairy sensor observations benchmark repository
* **Total Observations:** 1,059 rows
* **Total Features:** 7 independent variables + 1 target variable
* **Missing Values:** 0 null records across all columns
* **Target Variable (`Grade`):** `Low`, `Medium`, `High`

### Feature Definitions & Physical Significance

| Feature | Data Type | Empirical Range | Physical / Chemical Significance |
| :--- | :--- | :--- | :--- |
| **pH** | Continuous Float | 3.0 – 9.5 | Indicator of acidity/alkalinity. Fresh milk has a pH of 6.5–6.7. Values < 6.4 indicate lactic acid fermentation (souring); values > 6.8 indicate mastitis or adulteration. |
| **Temperature** | Continuous Float | 34.0°C – 90.0°C | Thermal regime. Raw milk drawn at ~37°C. Storage above 40°C triggers rapid microbial multiplication. |
| **Taste** | Binary Flag | 0 or 1 | Sensory evaluation. `1` = optimal palatable milk taste; `0` = off-flavor or sourness. |
| **Odor** | Binary Flag | 0 or 1 | Olfactory check. `1` = pleasant fresh milk aroma; `0` = pungent, sour, or absent aroma. |
| **Fat** | Binary Flag | 0 or 1 | Lipid composition. `1` = optimal standard dairy lipid concentration; `0` = skimmed or deficient fat. |
| **Turbidity** | Binary Flag | 0 or 1 | Light scattering caused by colloidal casein micelles and fat globules. `1` = standard turbid appearance; `0` = watery/separated milk. |
| **Colour** | Discrete Integer | 240 – 255 | Spectrophotometric reflectance index. `255` represents pure white milk; lower values represent discoloration or yellowing. |
| **Grade (Target)** | Categorical | Low, Medium, High | Quality grade defining marketability and consumption safety. |

---

## 4. Data Cleaning & The Duplicate Paradox

### The Discovery
* **Raw Observations Ingested:** 1,059 rows
* **Exact Duplicate Rows:** 976 rows (**92.16%**)
* **Unique Physical Milk Profiles:** **83 records**

### The Data Leakage Paradox
Widely circulated tutorials on this dataset apply `train_test_split` directly on the 1,059 rows without deduplication. Because 92.16% of rows are repeated, identical sensor combinations end up in both training and testing partitions. Models achieve an artificial **99.5% – 100% test accuracy** through verbatim memorization.

### Preprocessing Remediation
1. Stripped whitespace from column headers (e.g. `'Fat '` → `'Fat'`).
2. Standardized spelling variations (`'Temprature'` → `'Temperature'`).
3. Capitalized target labels (`'low'` → `'Low'`).
4. **Deduplication:** Isolated the 83 distinct physical samples (`Medium`: 34, `Low`: 26, `High`: 23).
5. **Stratified Split:** 80% (66 samples) for cross-validation and training; 20% (17 samples) strictly held out for testing.
6. **Leakage-Free Scaling:** `StandardScaler` fitted **strictly on X_train** numerical columns (`pH`, `Temperature`, `Colour`) and applied downstream.

---

## 5. Exploratory Data Analysis (EDA)

All 10 academic figures are generated by `src/eda.py` and saved to `reports/figures/`:

1. `grade_distribution.png`: Target class distribution across quality grades.
2. `ph_distribution.png`: Histogram & KDE showing normal pH concentration around 6.6 with acidic/alkaline extremes.
3. `temperature_distribution.png`: Bimodal thermal distribution with peaks at storage (~37°C) and abuse (>50°C).
4. `colour_distribution.png`: Reflectance distribution heavily concentrated at index 255.
5. `boxplot_ph_grade.png`: High quality milk is strictly bounded within pH 6.5–6.8; Low quality exhibits wide dispersion (3.0–9.5).
6. `boxplot_temperature_grade.png`: High quality milk is strictly capped at ≤ 45°C; Low quality reaches up to 90°C.
7. `boxplot_colour_grade.png`: Reflectance variations across grades.
8. `feature_distribution_plots.png`: Multi-panel 7-feature distribution grid.
9. `correlation_heatmap.png`: Pearson correlation matrix showing negative correlation between temperature and quality ($r = -0.49$).
10. `feature_vs_quality.png`: Binary feature proportions (High grade milk exhibits 99.6% optimal fat and 75% optimal odor).

---

## 6. Model Benchmarking & Experimental Results

All models were evaluated under **5-Fold Stratified Cross-Validation** on the training partition (`X_train`, 66 samples), followed by evaluation on the untouched holdout test partition (`X_test`, 17 samples) using `random_state=42`.

| Model | CV Mean Accuracy | CV Std | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | **0.8341** | ±0.1003 | **0.8824** | **0.8995** | **0.8824** | **0.8723** |
| **Random Forest** | **0.8473** | ±0.0982 | **0.8235** | **0.8505** | **0.8235** | **0.8188** |
| **Support Vector Machine (SVM)** | **0.8791** | ±0.1038 | **0.8235** | **0.8497** | **0.8235** | **0.8162** |
| **Decision Tree** | 0.8165 | ±0.1350 | 0.7647 | 0.8693 | 0.7647 | 0.7582 |
| **Logistic Regression** | 0.7868 | ±0.0913 | 0.6471 | 0.7132 | 0.6471 | 0.6434 |

### Final Model Selection
* **Selected Best Model:** **K-Nearest Neighbors** (k=3, distance weighting, euclidean metric).
* **Selection Criterion:** Achieved highest holdout test accuracy (88.24%), highest weighted F1-score (87.23%), and strong cross-validation stability (83.41% ± 10.03%).
* **Runner-Up:** Random Forest (CV: 84.73%, Test Acc: 82.35%, F1: 81.88%).

---

## 7. Error Analysis

On the 17 holdout test observations, our selected model correctly classified 15 samples and produced only 2 boundary misclassifications:

1. **Observation 1:** `pH=6.5`, `Temp=37°C`, `Taste=0`, `Odor=1`, `Fat=1`, `Turbidity=1`, `Colour=245`.
   - *Actual:* Low | *Predicted:* High
   - *Cause:* Standard pH (6.5) and optimal temperature (37°C) mimic High grade profiles; however, off-taste (0) and lower reflectance (245) demoted ground truth to Low.
2. **Observation 2:** `pH=6.8`, `Temp=50°C`, `Taste=0`, `Odor=0`, `Fat=1`, `Turbidity=0`, `Colour=255`.
   - *Actual:* Low | *Predicted:* Medium
   - *Cause:* Elevated temperature (50°C) is near the decision threshold between Medium and Low grades.

---

## 8. Streamlit Web Application

The interactive web interface is implemented in `app.py`:

* **Real-Time Quality Inference:** Input sliders and selectors with empirical validation bounds and demo presets.
* **Confidence Distribution:** Real-time probability progress bars for Low, Medium, and High quality.
* **Parameter Verification Summary:** Displays submitted sensor inputs on a verification badge.
* **EDA Dashboard:** Interactive viewer for all 10 dataset distributions, correlation matrices, and statistics.
* **Model Benchmarking:** Live metric comparison tables, confusion matrices, and feature importance bar charts.
* **Leakage Paradox Showcase:** Interactive educational module explaining duplicate records and data leakage for academic examiners.

---

## 9. Project Directory Structure

```
Food Quality Analysis/
│
├── data/
│   └── milk_quality.csv             # 1,059 raw milk sensor observations
│
├── notebooks/
│   └── milk_quality_analysis.ipynb  # Executed, end-to-end Jupyter Notebook
│
├── src/
│   ├── preprocessing.py             # Data cleaning, scaling, and deduplication
│   ├── train.py                     # 5-model training, 5-fold CV, and artifact export
│   ├── evaluate.py                  # Confusion matrices, diagnostics, error analysis
│   ├── predict.py                   # Standalone inference class & CLI tool
│   └── eda.py                       # High-resolution generation of all 10 EDA figures
│
├── models/
│   ├── best_model.pkl               # Serialized production classifier
│   ├── all_models.pkl               # Serialized dictionary of all 5 candidate models
│   ├── scaler.pkl                   # Fitted StandardScaler on training data
│   └── test_data.pkl                # Test split and metadata for evaluation
│
├── reports/
│   ├── figures/                     # 20 publication-quality charts & confusion matrices
│   ├── model_results.csv            # Empirical benchmark comparison table
│   ├── leakage_comparison.csv       # Naive vs deduplicated performance table
│   └── error_analysis.csv           # Detailed inspection of misclassified test samples
│
├── app.py                           # Full-featured Streamlit Web Application
├── requirements.txt                 # Exact, verified dependencies
├── README.md                        # Project documentation
├── REPORT.md                        # 20-chapter comprehensive academic case-study report
└── .gitignore                       # Python, cache, and OS ignore rules
```

---

## 10. Installation & Execution

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <repository_url>
cd "Food Quality Analysis"
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Training & Evaluation Pipeline
```bash
# Clean data, train 5 models with 5-fold CV, and save artifacts
python src/train.py

# Generate all 10 EDA figures
python src/eda.py

# Generate confusion matrices, feature importances, and error diagnostics
python src/evaluate.py
```

### Step 4: Run CLI Inference
```bash
python src/predict.py --ph 6.6 --temperature 37 --taste 1 --odor 0 --fat 1 --turbidity 0 --colour 255
```

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Limitations & Future Scope

* **Dataset Scale:** 83 unique sensor combinations form the baseline; expanding to thousands of longitudinal dairy farm observations will further refine edge-case boundaries.
* **Continuous Sensory Inputs:** Taste, odor, fat, and turbidity are currently represented as binary indicators (0 or 1). Integrating continuous sensors (turbidity NTU, fat Gerber spectrometry, electronic nose sensors) will improve granularity.
* **Hardware Edge Integration:** Future work involves flashing the quantized model onto microcontrollers (ESP32 / Raspberry Pi) for direct inline quality checking on automated dairy milking pipelines.
