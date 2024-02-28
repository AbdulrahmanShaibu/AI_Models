
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV , cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# Dataset path
dataset_path = 'C:\\Users\\abdul\\Desktop\\ThirdYear\\AI\\heart+disease\\ClecelandDataset.xlsx'
# Read the Excel file
df = pd.read_excel(dataset_path)
# Write the DataFrame to an Excel file
df.to_excel(dataset_path, index=False)
# # Explore the dataset
print(df.info())
# #describing dataset statistics
print(df.describe())
# #Check for missing values in dataset
print(df.isnull().sum())
# #printing 1st 5 rows
print(df.head())

# # Print the column names and check for the presence of 'target'
print(df.columns)
# #printing last 5 rows
print(df.tail)
# #printing dataset shape
print(df.shape)

# #checking distribution values for target...0,1,2,3,4
print(df['Class Attribute'].value_counts())

# #Preprocessing
# #  the target variable is 'Class Attribute'
X = df.drop('Class Attribute', axis=1)
y = df['Class Attribute']

print(X)
print(y)

# # Split the dataset
X_train, X_test, y_train, y_test =train_test_split( X, y , test_size=0.2,
random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Step 1: Create a logistic regression model
model_data = LogisticRegression()

# Step 2: Perform 10-fold cross-validation
cv_scores = cross_val_score(model_data, X_train, y_train, cv=10, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))

# Step 3: Perform RandomizedSearchCV for hyperparameter tuning
#param_dist is a dictionary defining a set of hyperparameter values for the logistic regression model. These values will be explored during the random search for hyperparameter tuning, allowing the algorithm to find the combination of hyperparameters that results in the best performance on the given dataset.
#'solver': ['liblinear']: This line specifies the solver algorithm to be used in the logistic regression model. 'liblinear' is a solver suitable for small datasets and is capable of handling both L1 and L2 penalties.
# Logistic regression supports L1 (Lasso) and L2 (Ridge) regularization penalties
param_dist = {
    'C': np.logspace(-4, 4, 20),  # Regularization parameter
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

random_search = RandomizedSearchCV(model_data, param_distributions=param_dist, n_iter=100, cv=10, scoring='accuracy', random_state=42)
#It will try different combinations of hyperparameters and cross-validate each combination to find the best set of hyperparameters
random_search.fit(X_train, y_train)

# Prints the best hyperparameters found during the RandomizedSearchCV process.
print("Best parameters:", random_search.best_params_)
print("Best accuracy:", random_search.best_score_)

# Step 4: Retrieving the best model found during the hyperparameter tuning process
best_model = random_search.best_estimator_

# Step 5: Using the best model to make predictions on the training data.
predict_X_train = best_model.predict(X_train)

# Step 6: Calculating the accuracy of the model's predictions on the training data.
accuracy_prediction = accuracy_score(predict_X_train, y_train)
#average='weighted' parameter means that it computes the weighted average of the F1-score across different classes
f1_prediction = f1_score(predict_X_train, y_train, average='weighted')  # or average='macro'

print('Model accuracy:', accuracy_prediction)
print('Model F1-score:', f1_prediction)

# Step 7: Creating a confusion matrix
#The confusion matrix is a table that is often used to evaluate the performance of a classification algorithm
conf_matrix = confusion_matrix(y_train, predict_X_train)

# Step 8: Visualizing the confusion matrix
#createing a new figure for the plot with a specified size of 8 inches by 6 inches.
plt.figure(figsize=(8, 6))

#heatmap is created using the Seaborn library (sns)
#The annot=True parameter adds the actual numerical values in each cell of the heatmap
#The fmt="d" parameter specifies that the values should be displayed as integer
#cmap="Blues" parameter sets the color map to shades of blue.
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

