# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 08:58:23 2024

@author: Admin
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the data
# Assuming you have a CSV file with columns like 'GPA', 'Age', 'Gender', 'Enrollment_Status', 'Graduation_Status', etc.
data = pd.read_csv('/content/student data.csv')

# Select relevant features and target variable
features = ['GPA', 'Age', 'Gender']  # Add other relevant features as needed
target_enrollment = 'Enrollment_Status'
target_graduation = 'Graduation_Status'

# Convert categorical variables to numeric if necessary (e.g., Gender)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})  # Example mapping

# Create feature matrix (X) and target vector (y) for enrollment
X_enrollment = data[features]
y_enrollment = data[target_enrollment]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_enrollment, y_enrollment, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot ROC curve (Receiver Operating Characteristic)
y_prob = logreg_model.predict_proba(X_test_scaled)[:, 1]
y_test_numeric = y_test.apply(lambda x: 1 if x == 'Enrolled' else 0)
fpr, tpr, thresholds = roc_curve(y_test_numeric, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Optional: Feature importance (coefficients of the logistic regression model)
coefficients = logreg_model.coef_[0]
feature_importance = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
feature_importance['Importance'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance in Logistic Regression')
plt.grid(axis='x')
plt.show()