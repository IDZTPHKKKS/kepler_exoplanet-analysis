# Exoplanet Disposition Prediction using Kepler Data

This project aims to predict the disposition of exoplanets observed by NASA’s Kepler mission using machine learning techniques. The dataset used in this project contains several key features related to the exoplanets and their host stars. 

## Dataset Description

The features in the dataset include:

- **Orbital Period (koi_period)**: The number of days it takes for an exoplanet to complete one orbit around its star.
- **Exoplanet Radius (koi_prad)**: The radius of the exoplanet in Earth radii.
- **Star Radius (koi_srad)**: The radius of the star around which the exoplanet orbits, in solar radii.
- **Star Temperature (koi_steff)**: The effective temperature of the host star, in Kelvin.
- **Signal-to-Noise Ratio (koi_model_snr)**: A measure of the data quality; higher values represent clearer data.
- **Target Variable (koi_disposition)**: This variable indicates the disposition of the exoplanet. It has three possible values:
  - 0: **False Positive** - The exoplanet is a false detection.
  - 1: **Candidate** - The exoplanet is a potential candidate for further investigation.
  - 2: **Confirmed** - The exoplanet is a confirmed detection.

## Data Preprocessing

The dataset undergoes several preprocessing steps before being fed into the machine learning model:

1. **Missing Value Handling**: 
   - Columns with more than 50% missing data are dropped.
   - Missing values in numerical columns are imputed using the median value of the column.
   
2. **Feature Scaling**: 
   - All features are normalized using `StandardScaler` to ensure they are on a similar scale, improving model performance.
   
3. **Handling Class Imbalance**: 
   - The dataset is imbalanced, so **SMOTE (Synthetic Minority Over-sampling Technique)** is used to generate synthetic samples for the under-represented classes.

## Model and Evaluation

The machine learning model used in this project is the **RandomForestClassifier**, which is a powerful ensemble learning method. To optimize the model, **GridSearchCV** is used to perform hyperparameter tuning, including the number of trees in the forest, the maximum depth of trees, and the minimum number of samples required to split a node.

The model's performance is evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The percentage of true positives among all predicted positives.
- **Recall**: The percentage of true positives among all actual positives.
- **F1-score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix showing the true positives, false positives, true negatives, and false negatives.

- ## Data Visualization

To better understand the dataset and its features, the following visualizations were created:

- **Classification Report**: This image summarizes the precision, recall, and F1-scores for each class, as well as the overall accuracy.
- ![image](https://github.com/user-attachments/assets/625c816c-03d9-43ea-b1a4-d1c43c2add0d)

- **Correlation Matrix**: This heatmap shows the correlation between all features in the dataset, providing insights into multicollinearity and feature relationships.
-![image](https://github.com/user-attachments/assets/60443e80-bf02-499f-9499-24e131ae50ef)


- **Confusion Matrix**: This visualization highlights the performance of the model by showing the counts of true positives, false positives, true negatives, and false negatives for each class.
- ![image](https://github.com/user-attachments/assets/e8b4ff52-102b-4a40-9952-136e25075292)

- **Feature Importances**: This bar chart shows the relative importance of each feature in the model, helping to identify the key drivers of the predictions.
- ![image](https://github.com/user-attachments/assets/963ee4b1-294e-4441-aeb8-e8fc5d544d8c)

---

## Project Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/exoplanet-disposition-prediction.git

