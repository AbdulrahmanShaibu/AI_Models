import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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

print('_________________*************************__________________*******************')
print('USING RANDOM FOREST ALGORITHM FOR A MODEL')

# Step 1: Create a Random Forest model
model_data_rf = RandomForestClassifier()

# Step 2: Perform 10-fold cross-validation
cv_scores_rf = cross_val_score(model_data_rf, X_train, y_train, cv=10, scoring='accuracy')

# Print the cross-validation scores for Random Forest
print("Cross-validation scores for Random Forest:", cv_scores_rf)
print("Mean accuracy for Random Forest:", np.mean(cv_scores_rf))

# Step 3: Perform RandomizedSearchCV for hyperparameter tuning for Random Forest
param_dist_rf = {
    'n_estimators': [10, 20, 50, 100, 150, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap sampling  are used when building trees
}

random_search_rf = RandomizedSearchCV(model_data_rf, param_distributions=param_dist_rf, n_iter=100, cv=10, scoring='accuracy', random_state=42)
random_search_rf.fit(X_train, y_train)

# Print the best parameters and their corresponding accuracy for Random Forest
print("Best parameters for Random Forest:", random_search_rf.best_params_)
print("Best accuracy for Random Forest:", random_search_rf.best_score_)

# Step 4: Use the best model from RandomizedSearchCV for Random Forest
best_model_rf = random_search_rf.best_estimator_

# Step 5: Predict using the best Random Forest model
predict_X_train_rf = best_model_rf.predict(X_train)

# Step 6: Calculate accuracy and F1-score for Random Forest
accuracy_prediction_rf = accuracy_score(predict_X_train_rf, y_train)
f1_prediction_rf = f1_score(predict_X_train_rf, y_train, average='weighted')  # or average='macro'

# Step 7: Calculate and print the confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_train, predict_X_train_rf)
print('Confusion Matrix for Random Forest:')
print(conf_matrix_rf)

print('Random Forest Model accuracy:', accuracy_prediction_rf)
print('Random Forest Model F1-score:', f1_prediction_rf)

conf_matrix = confusion_matrix(y_train, predict_X_train_rf)
# Step 8: Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()