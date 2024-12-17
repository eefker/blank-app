@@ -1,19 +1,109 @@
# 🎈 Blank app template
# Framingham Heart Study - BMI and CHD Prediction (Phase 2)

A simple Streamlit app template for you to modify!
## This repository contains an interactive Python-based application and accompanying notebook for analyzing and predicting the influence of Body Mass Index (BMI) on the prevalence of Coronary Heart Disease (CHD). The analysis is based on the Framingham Heart Study dataset. The project was born for the second phase of Introduction to Programming Python (course code: MAI3002) at University of Maastricht. 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)
## Dataset
The dataset used in this project is sourced from the Framingham Heart Study, a longitudinal study focusing on cardiovascular disease. It includes various features such as:
Age, BMI, Cholesterol levels, Blood Pressure, Smoking status, etc.
The data file is loaded dynamically from the following URL:
Framingham Dataset on GitHubDataset (https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv)
### Objective
Investigate the relationship between Body Mass Index (BMI) and the prevalence of Coronary Heart Disease (CHD).
Use machine learning models to build predictive models for assessing CHD risk.
## Files Included
szegedi,m_rutten,e_(i6274290)_project_mai3002_phase2.py
The main Python script containing:
Data loading, exploration, and cleaning.
Feature engineering and preparation.
Model building: Logistic Regression, Support Vector Machine, Random Forest, Neural Networks, etc.
Model evaluation using metrics such as accuracy, confusion matrix, and AUC-ROC.
Visualizations for better understanding and explanation.
Jupyter Notebook (optional, if provided):
A notebook version of the Python script for detailed explanations and visualizations in a cell-based format.
README.md
Documentation describing the code, how to use it, and setup instructions.
## Installation
### 1. Prerequisites
Ensure the following tools and libraries are installed on your system:
Python 3.8+
Streamlit
Pandas
Matplotlib
Seaborn
Scikit-learn
TensorFlow/Keras (because neural networks are included)
### 2. Install the requirements
Install the required libraries using the command below:
pip install streamlit pandas matplotlib seaborn scikit-learn tensorflow keras
git clone https://github.com/yourusername/bmi-chd-prediction.git
cd bmi-chd-prediction

### How to run it on your own machine
streamlit run szegedi,m_rutten,e_(i6274290)_project_mai3002_phase2.py

1. Install the requirements
## Usage

   ```
   $ pip install -r requirements.txt
   ```
### Introduction
Provides an overview of the research question and objectives.
### Data Exploration and Cleaning
Displays data statistics, handles missing values, and visualizes data.
### Model Building
Allows the selection of predictor variables and machine learning models (e.g., Logistic Regression, SVM, Random Forest, Neural Network).
Displays training results, model accuracy, and confusion matrices.
### Model Evaluation
Evaluates model performance using metrics such as accuracy, AUC-ROC, and confusion matrices.

2. Run the app
## Features

   ```
   $ streamlit run streamlit_app.py
   ```
Interactive UI: Use checkboxes, dropdown menus, and sliders to interact with the data and models.
Dynamic Visualization: Heatmaps, scatter plots, and confusion matrices.
Model Performance Metrics: Accuracy, AUC-ROC, F1 Score, and more.
Machine Learning Models:
Logistic Regression
Support Vector Machine (SVM)
Decision Trees
Random Forest
Neural 
## Results
### The results provide insights into the role of BMI as a predictor for CHD and include:
Visualizations for BMI categories and CHD prevalence.
Comparative model performance metrics.
Neural network architecture visualizations.
## Contributing
### If you'd like to contribute:
Fork this repository.
Create a feature branch: git checkout -b feature-branch-name.
Commit your changes: git commit -m "Add a new feature".
Push to the branch: git push origin feature-branch-name.
Submit a pull request.
## License
This project is licensed under the UM License.
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)