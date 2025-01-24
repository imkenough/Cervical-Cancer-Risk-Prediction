---

# Cervical Cancer Risk Prediction

This repository contains the implementation and documentation for the **Cervical Cancer Risk Prediction** project. The goal of this project is to develop a machine learning model that provides precise risk assessments for cervical cancer, leveraging demographic, clinical, and biological data. This work has significant implications for early detection, prevention, and optimized healthcare resource allocation.
Sophomore Year Machine Learning course project.

_this is a condensed readme file; a full project report and presentation are part of this repo_

[Project Report](Project%20Report-%20Cervical%20Cancer%20Risk%20Predictionc.pdf) <br>
[Presentation](Cervical%20Cancer%20Detection%20Using%20Machine%20Learning.pdf)


---

## Table of Contents

1. [Introduction](#introduction)
2. [Features and Methodology](#features-and-methodology)
3. [Results](#results)
4. [Installation and Usage](#installation-and-usage)
5. [Future Work](#future-work)
6. [Contributors](#contributors)
7. [References](#references)

---

## Introduction

Cervical cancer is one of the most preventable yet prevalent cancers affecting women globally, especially in areas with limited healthcare access. This project employs **XGBoost**, a robust machine learning algorithm, to predict cervical cancer risk with high accuracy. By analyzing key risk factors such as age, sexual behavior, smoking habits, and HPV infection status, the model provides personalized risk assessments to support healthcare providers in tailoring preventive strategies.

### Key Objectives:
- Enhance early detection of cervical cancer.
- Improve resource allocation for screening programs.
- Reduce the global burden of cervical cancer.

---

## Features and Methodology

### Key Features:
- **Algorithm**: XGBoost, optimized for gradient-boosted decision trees.
- **Data Preprocessing**: Handled missing values, normalized numerical features, and encoded categorical variables.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Area Under the ROC Curve (AUC-ROC)
  - Confusion Matrix

### Methodology:
1. **Data Preparation**: Preprocessed the dataset to handle missing and categorical data.
2. **Model Training**:
   - Used an 80:20 train-test split.
   - Hyperparameter tuning for optimal performance.
3. **Performance Evaluation**: Compared XGBoost with Logistic Regression, Random Forest, and Support Vector Machines.

### Why XGBoost?
- Superior handling of non-image data.
- High accuracy and robust performance metrics.
- Effective regularization techniques to prevent overfitting.

---

## Results

The XGBoost model demonstrated exceptional performance:
- **Accuracy**: 99.04%
- **Precision**: 90.91%
- **Recall (Sensitivity)**: 100%
- **F1 Score**: 95%
- **AUC-ROC**: 99.26%

These results highlight the model's capability to provide accurate risk assessments and its potential for clinical application.

---

## Installation and Usage

### Prerequisites:
- Python 3.8 or higher
- Required libraries: `xgboost`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cervical-cancer-risk-prediction.git
   cd cervical-cancer-risk-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage:
1. Preprocess the dataset:
   - Place the dataset in the `data/` directory.
   - Run the preprocessing script:
     ```bash
     python preprocess.py
     ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---

## Future Work

1. **Clinical Integration**: Embed the model into healthcare workflows for real-time risk assessment.
2. **Advanced Features**: Incorporate additional risk factors and explore deep learning techniques.
3. **Health Equity**: Address disparities by optimizing the model for underserved populations through telemedicine platforms.

---

## Contributors

- **Abhijith Nair**
- **Haneesh Kenny**
- **Aditya Vats**
- **Mehul Karwa**

---

## References

1. [A Model for Predicting Cervical Cancer Using Machine Learning Algorithms](https://www.researchgate.net/publication/360969531)
2. [Cervical Cancer Prediction Using Support Vector Machines](https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0295632)
3. [Risk Stratification for Cervical Cancer Detection Using Machine Learning Classifiers](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8733205/)

---
