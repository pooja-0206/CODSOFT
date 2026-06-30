# ============================================================
# TASK 1: TITANIC SURVIVAL PREDICTION
# CodSoft Data Science Internship
# ============================================================
# This script predicts whether a passenger survived the Titanic
# using a Random Forest Classifier.
# ============================================================

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   TITANIC SURVIVAL PREDICTION - CodSoft Internship")
print("=" * 55)

# ============================================================
# --- Step 2: Load Dataset ---
# ============================================================
# Download the Titanic dataset from:
# https://www.kaggle.com/datasets/yasserh/titanic-dataset
# Save it as 'Titanic-Dataset.csv' in the same folder as this script.

df = pd.read_csv('Titanic-Dataset.csv', encoding='latin1')

print("\n[1] First 5 rows of the dataset:")
print(df.head())

print("\n[2] Dataset Shape (rows, columns):", df.shape)

print("\n[3] Dataset Info:")
print(df.info())

print("\n[4] Missing Values in each column:")
print(df.isnull().sum())

# ============================================================
# --- Step 3: Data Preprocessing ---
# ============================================================

# Drop columns that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Fill missing Age values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the most common port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert 'Sex' column: male -> 0, female -> 1
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Convert 'Embarked' column: C -> 0, Q -> 1, S -> 2
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

print("\n[5] Data after preprocessing (first 5 rows):")
print(df.head())

print("\n[6] Missing Values after cleaning:")
print(df.isnull().sum())

# ============================================================
# --- Step 4: Exploratory Data Analysis (EDA) ---
# ============================================================

# Survival count plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Count (0 = Did Not Survive, 1 = Survived)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('survival_count.png')
plt.show()
print("\n[7] Survival count plot saved as 'survival_count.png'")

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Gender (0 = Male, 1 = Female)')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('survival_by_gender.png')
plt.show()
print("[8] Survival by gender plot saved as 'survival_by_gender.png'")

# Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set3')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('survival_by_class.png')
plt.show()
print("[9] Survival by class plot saved as 'survival_by_class.png'")

# Age Distribution
plt.figure(figsize=(8, 4))
df['Age'].hist(bins=30, color='steelblue', edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.show()
print("[10] Age distribution plot saved as 'age_distribution.png'")

# ============================================================
# --- Step 5: Feature Selection ---
# ============================================================

# X = features (inputs), y = target (what we want to predict)
X = df.drop(columns=['Survived'])
y = df['Survived']

print("\n[11] Features used for training:", list(X.columns))

# ============================================================
# --- Step 6: Train-Test Split ---
# ============================================================
# 80% data for training, 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[12] Training samples : {X_train.shape[0]}")
print(f"     Testing  samples : {X_test.shape[0]}")

# ============================================================
# --- Step 7: Model Training ---
# ============================================================

# --- Model 1: Logistic Regression ---
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)

# --- Model 2: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print("\n[13] Model Accuracies:")
print(f"     Logistic Regression : {lr_acc * 100:.2f}%")
print(f"     Random Forest       : {rf_acc * 100:.2f}%")

# ============================================================
# --- Step 8: Evaluation ---
# ============================================================

print("\n[14] Random Forest - Classification Report:")
print(classification_report(y_test, rf_preds,
                             target_names=['Did Not Survive', 'Survived']))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("[15] Confusion matrix saved as 'confusion_matrix.png'")

# ============================================================
# --- Step 9: Feature Importance ---
# ============================================================

feature_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
feature_importances.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("[16] Feature importance plot saved as 'feature_importance.png'")

print("\n[17] Feature Importances:")
print(feature_importances)

# ============================================================
# --- Step 10: Predict for a New Passenger ---
# ============================================================

# Example: Predict survival for a new passenger
# [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
# Sex: 0=male, 1=female | Embarked: 0=C, 1=Q, 2=S

new_passenger = pd.DataFrame({
    'Pclass':   [3],       # 3rd class
    'Sex':      [0],       # Male
    'Age':      [22],      # 22 years old
    'SibSp':    [1],       # 1 sibling/spouse aboard
    'Parch':    [0],       # No parents/children
    'Fare':     [7.25],    # Fare paid
    'Embarked': [2]        # Embarked from Southampton
})

prediction = rf_model.predict(new_passenger)
result = "SURVIVED ✓" if prediction[0] == 1 else "DID NOT SURVIVE ✗"

print("\n" + "=" * 55)
print("   PREDICTION FOR NEW PASSENGER")
print("=" * 55)
print(f"   Passenger Details : 3rd class, 22-year-old male")
print(f"   Prediction        : {result}")
print("=" * 55)

print("\n✅ Task 1 Complete! All plots saved to your folder.")
print("   Push this file and the plots to your CODSOFT GitHub repo.")
