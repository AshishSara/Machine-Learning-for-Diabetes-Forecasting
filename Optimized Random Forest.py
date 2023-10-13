# Import libraries
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from lime.lime_tabular import LimeTabularExplainer
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Function for data preprocessing
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    return X_resampled, y_resampled


# Function for training and hyperparameter tuning
def train_and_tune(X_train, y_train):
    random_forest_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]}
    grid_search_rf = GridSearchCV(random_forest_clf, param_grid_rf, cv=3, scoring='recall', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    return grid_search_rf.best_estimator_


# Function for model evaluation
def evaluate_model(clf, X_test, y_test, name):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Performance of {name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix: \n{conf_matrix}")
    print("=" * 30)

#function to interpret with Lime
def interpret_with_lime(model, X_test):
    explainer = LimeTabularExplainer(X_test, feature_names=X.columns.tolist(), class_names=['No Diabetes', 'Diabetes'],
                                     mode='classification')
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(X_test[i], model.predict_proba)
    exp.show_in_notebook()


# Function to interpret with SHAP
def interpret_with_shap(model, X_train, X_test):
    # Initialize JavaScript visualizations for SHAP
    shap.initjs()

    # Create the SHAP explainer object
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for the test dataset
    shap_values = explainer.shap_values(X_test)

    # Visualize the SHAP values for a single prediction
    i = np.random.randint(0, X_test.shape[0])
    shap.force_plot(explainer.expected_value[1], shap_values[1][i, :], X_test[i, :], feature_names=X.columns.tolist())


# Main code execution starts here
# Fetch and load the dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets.to_numpy().ravel()

# Preprocess data
X_resampled, y_resampled = preprocess_data(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and tune the model
best_rf = train_and_tune(X_train, y_train)

# Evaluate the model
evaluate_model(best_rf, X_test, y_test, 'Optimized Random Forest')

# Evaluate the model
evaluate_model(best_rf, X_test, y_test, 'Optimized Random Forest')

# Interpret a single prediction with LIME
interpret_with_lime(best_rf, X_test)

# Interpret a single prediction with SHAP
interpret_with_shap(best_rf, X_train, X_test)

# Plot feature importances
features = X.columns.tolist()
importances = best_rf.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()
