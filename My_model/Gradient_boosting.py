
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import  GradientBoostingClassifier
import os

# Dataset path
dataset_path = 'C:\\Users\\abdul\\Desktop\\ThirdYear\\AI\\heart+disease\\ClecelandDataset.xlsx'
# Read the Excel file
df = pd.read_excel(dataset_path)
# Write the DataFrame to an Excel file
df.to_excel(dataset_path, index=False)

# Explore the dataset
print(df.info())
#describing dataset statistics
print(df.describe())
#Check for missing values in dataset
print(df.isnull().sum())
#printing 1st 5 rows
print(df.head())

# Print the column names and check for the presence of 'target'
print(df.columns)
#printing last 5 rows
print(df.tail)
#printing dataset shape
print(df.shape)

#checking distribution values for target...0,1,2,3,4
print(df['Class Attribute'].value_counts())

#Preprocessing
#  the target variable is 'Class Attribute'
X = df.drop('Class Attribute', axis=1)
y = df['Class Attribute']

print(X)
print(y)

# Split the dataset
X_train, X_test, y_train, y_test =train_test_split( X, y , test_size=0.2,
random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

print('____********USING GRADIENT BOOSTING****____')
# Step 1: Create a Gradient Boosting model
model_data = GradientBoostingClassifier()

# Step 2: Perform 10-fold cross-validation
cv_scores = cross_val_score(model_data, X_train, y_train, cv=10, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))

# Step 3: Perform RandomizedSearchCV for hyperparameter tuning
param_dist = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(model_data, param_distributions=param_dist, n_iter=100, cv=10, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Print the best parameters and their corresponding accuracy
print("Best parameters:", random_search.best_params_)
print("Best accuracy:", random_search.best_score_)

# Step 4: Use the best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Step 5: Predict using the best model
predict_X_train = best_model.predict(X_train)

# Step 6: Calculate accuracy and F1-score
accuracy_prediction = accuracy_score(predict_X_train, y_train)
f1_prediction = f1_score(predict_X_train, y_train, average='weighted')  # or average='macro'

print('Model accuracy:', accuracy_prediction)
print('Model F1-score:', f1_prediction)

conf_matrix = confusion_matrix(y_train, predict_X_train)
# Step 8: Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()