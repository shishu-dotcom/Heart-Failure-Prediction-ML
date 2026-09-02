# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, accuracy_score
)
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier

# Load dataset
file_path = 'saved_dataset.csv'
dataset = pd.read_csv(file_path)

# Data preprocessing
# Separate numerical and categorical columns
numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = dataset.select_dtypes(include=['object']).columns

# Handle missing values for numerical columns using median imputation
numerical_imputer = SimpleImputer(strategy='median')
dataset[numerical_cols] = numerical_imputer.fit_transform(dataset[numerical_cols])

# Handle missing values for categorical columns using the most frequent strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

# Encode categorical features and target variable
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Split features and target
X = dataset.drop('Heart Disease', axis=1).values
y = dataset['Heart Disease'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize TabNetClassifier
tabnet_clf = TabNetClassifier()

# Train TabNet model
tabnet_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=['test'],
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=10,
    batch_size=256,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

# Make predictions
y_pred = tabnet_clf.predict(X_test)
y_proba = tabnet_clf.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Feature Importance
feature_importances = tabnet_clf.feature_importances_
plt.figure()
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()
