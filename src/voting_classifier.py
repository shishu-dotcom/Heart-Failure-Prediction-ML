# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, accuracy_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

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
X = dataset.drop('Heart Disease', axis=1)
y = dataset['Heart Disease']

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create base models
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('decision_tree', dt), ('random_forest', rf), ('logistic_regression', lr)],
    voting='soft'  # Use 'hard' for majority voting
)

# Train Voting Classifier
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:, 1]

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

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Voting Classifier')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

# F1 Score Curve
f1_scores = 2 * (precision * recall) / (precision + recall)
plt.figure()
plt.plot(recall[:-1], f1_scores[:-1], label="F1 Score")
plt.xlabel("Recall")
plt.ylabel("F1 Score")
plt.title("F1 Score")
plt.legend()
plt.show()

# MCC Curve
thresholds = np.linspace(0, 1, 100)
mcc_scores = []
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-9)
    mcc_scores.append(mcc)

plt.figure()
plt.plot(thresholds, mcc_scores, label="MCC")
plt.xlabel("Probability")
plt.ylabel("MCC")
plt.title("MCC")
plt.legend()
plt.show()
