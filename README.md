# Online Shoppers Purchasing Intention Prediction

## Problem Statement

The objective of this project is to predict whether an online shopper will generate revenue (make a purchase) during a browsing session. The project uses machine learning classification algorithms to identify purchasing intent based on session-related features.

The trained models are made available through an interactive Streamlit application. Users can upload a test dataset, select a classification model, and view predictions and evaluation results.

## Dataset Description

The **Online Shoppers Purchasing Intention Dataset** was obtained from the UCI Machine Learning Repository.

- **Number of instances:** 12,330
- **Number of input features:** 17
- **Target variable:** `Revenue`
- **Problem type:** Binary classification

The target variable indicates whether a browsing session resulted in a purchase (`True`) or did not result in a purchase (`False`).

The dataset contains numerical and categorical features describing visitor behaviour and browsing sessions. Categorical features were encoded, and numerical features were scaled as part of preprocessing.

## GitHub Repository Link

**GitHub Repository:** 

The repository contains:
- `app.py` – Streamlit application.
- `requirements.txt` – Python dependencies required to run the application.
- `test_data.csv` – Test dataset used by the application.
- `models/` – Saved machine learning models.
- `model-training.ipynb`  Notebook containing the model training and evaluation
- `README.md` – Project documentation.

## d. Models Used

The dataset was split into training and testing sets using an 80:20 stratified split, with `random_state=42`.

Five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8500 | 0.8962 | 0.5107 | 0.7487 | 0.6072 | 0.5330 |
| Decision Tree | 0.8544 | 0.7033 | 0.5331 | 0.4843 | 0.5075 | 0.4230 |
| KNN | 0.8719 | 0.7785 | 0.6435 | 0.3874 | 0.4837 | 0.4331 |
| Gaussian Naive Bayes | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Support Vector Machine (SVM) | 0.8832 | 0.8861 | 0.6691 | 0.4869 | 0.5636 | 0.5065 |

### Observations

- **Logistic Regression:** Achieved the highest AUC, Recall, F1 Score, and MCC. It identifies a larger proportion of actual purchasing sessions and provides a strong balance between precision and recall.
- **Decision Tree:** Achieved moderate accuracy, but its AUC and F1 Score were comparatively lower than those of Logistic Regression and SVM.
- **KNN:** Achieved good accuracy and precision, but its recall was relatively low, meaning it missed a considerable number of actual purchasing sessions.
- **Gaussian Naive Bayes:** Achieved the highest recall, but its precision and accuracy were lower, resulting in more false-positive predictions.
- **SVM:** Achieved the highest accuracy and precision, along with strong AUC performance. Its predictions were comparatively reliable when identifying purchasing sessions.

### Overall Winner

**Support Vector Machine (SVM)** is selected as the preferred model for this project. Among the five classifiers, SVM achieved the highest accuracy of **0.8832** and precision of **0.6691**. It also demonstrated strong AUC performance of **0.8861** and an F1 Score of **0.5636**.

These results indicate that SVM provides reliable purchase predictions while maintaining good overall classification performance. Logistic Regression also performed well, achieving the highest AUC, Recall, F1 Score, and MCC. However, considering the higher accuracy and precision of SVM, it is selected as the preferred model for this application. Logistic Regression remains a strong alternative when recall and F1 Score are the priority.

## Application

The Streamlit application provides an interactive interface for using the trained classification models.

### Application features

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Predict purchase intention for uploaded records.
- Evaluate the selected model using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View the confusion matrix and classification report.
- Compare the performance of all five models.
- Download the prediction results.

The application loads the saved models from the `models/` directory. It does not retrain the models when a user uploads a test dataset.

## Deployment

The application is deployed using Streamlit Community Cloud.

**Live Streamlit Application:**
