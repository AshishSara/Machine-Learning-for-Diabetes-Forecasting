# Machine-Learning-for-Diabetes-Forecasting
Developed an optimized Random  Forest model to Predict Diabetes with high accuracy


The Diabetes Health Indicators Dataset is a comprehensive resource for studying the relationship between lifestyle and the prevalence of diabetes in the United States.
Below are key points about this dataset:

Overview
Purpose: To better understand the relationship between lifestyle and diabetes in the U.S.
Funding: Provided by the Centers for Disease Control and Prevention (CDC).
Subject Area: Life Science, specifically diabetes and associated lifestyle factors.
Associated Tasks: Primarily classification, given the target variables (diabetes, pre-diabetes, healthy).
Data Types: Categorical and Integer features.
Instances: 253,680
Features: 21
Data Splits: Either cross-validation or a fixed train-test split is recommended.
Sensitive Data: Contains potentially sensitive data like Gender, Income, and Education level.
Missing Values: No
Variables
Target Variable
Diabetes_binary: Binary classification indicating whether an individual has no diabetes (0) or has prediabetes/diabetes (1).
Feature Variables
HighBP, HighChol, CholCheck: Binary features related to blood pressure and cholesterol.
BMI: Integer feature for Body Mass Index.
Smoker, Stroke, HeartDiseaseorAttack: Binary features related to other health conditions.
PhysActivity, Fruits, Veggies: Binary features related to lifestyle choices.
HvyAlcoholConsump, AnyHealthcare, NoDocbcCost: Binary features related to health care and lifestyle.
GenHlth, MentHlth, PhysHlth: Integer features related to general, mental, and physical health.
DiffWalk: Binary feature indicating difficulty in walking or climbing stairs.
Sex: Binary feature for gender (0 for female, 1 for male).
Age: Integer feature with 13-level age categories.
Education, Income: Integer features for education level and income scale.
Context
The dataset helps in understanding the significant burden of diabetes, both in terms of public health and economic costs. Given the increasing prevalence of this chronic condition, the dataset can be valuable for predictive modeling, informing public health policies, and encouraging early diagnosis and effective treatment.

Source and Documentation
The data comes from CDC's Behavioral Risk Factor Surveillance System (BRFSS).
An introductory paper related to this dataset has been published in the Morbidity and Mortality Weekly Report.
Data Files
diabetes_012_health_indicators_BRFSS2015.csv: Original dataset with class imbalance.
diabetes_binary_5050split_health_indicators_BRFSS2015.csv: Balanced dataset with a 50-50 split of target classes.
diabetes_binary_health_indicators_BRFSS2015.csv: Another version with class imbalance but only two target classes.
Given its detailed features, this dataset can be a valuable resource for predictive modeling to assess diabetes risk based on various health indicators and demographic information.



Source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset | DOI: 10.24432/C53919



Three Models were built and evaluated to begin the process, here were the results:

Performance of Optimized Logistic Regression:
  Accuracy: 0.7551
  Precision: 0.7416
  Recall: 0.7811
  F1 Score: 0.7609
  Confusion Matrix:
True Positives = 31920, True Negatives = 11853, False Positives = 9535, False Negatives = 34026


Performance of Optimized Gradient Boosting:
  Accuracy: 0.8866
  Precision: 0.8926
  Recall: 0.8783
  F1 Score: 0.8854
  Confusion Matrix:
True Positives = 39172,  True Negatives = 4601, False Positives = 5303, False Negatives = 38258

Evaluation Metrics:
Accuracy: 0.9011
Precision: 0.8953
Recall: 0.9079
F1 Score: 0.9015
Confusion Matrix:
True Positives = 39547, True Negatives = 39149, False Positives = 4624, False Negatives = 4014
