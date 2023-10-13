# Import libraries
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Fetch and load the dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets.to_numpy().ravel()  # Convert to 1D array

# Data Preprocessing
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize classifiers
logistic_clf = LogisticRegression(random_state=42)
random_forest_clf = RandomForestClassifier(random_state=42)
gradient_boosting_clf = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning for Logistic Regression
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid_search_lr = GridSearchCV(logistic_clf, param_grid_lr, cv=3, scoring='recall', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_

# Hyperparameter tuning for Random Forest
'''
param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]}
grid_search_rf = GridSearchCV(random_forest_clf, param_grid_rf, cv=3, scoring='recall', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
'''
# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1]}
grid_search_gb = GridSearchCV(gradient_boosting_clf, param_grid_gb, cv=3, scoring='recall',n_jobs=-1)
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_

# Evaluate the best models
best_classifiers = [
    (best_lr, 'Optimized Logistic Regression'),
    #(best_rf, 'Optimized Random Forest'),
    (best_gb, 'Optimized Gradient Boosting')
]

# Train and evaluate models
for clf, name in best_classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display results
    print(f"Performance of {name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix: \n{conf_matrix}")
    print("=" * 30)
