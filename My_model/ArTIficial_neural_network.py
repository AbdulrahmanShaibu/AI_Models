import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Dataset path
dataset_path = 'C:\\Users\\abdul\\Desktop\\ThirdYear\\AI\\heart+disease\\ClecelandDataset.xlsx'

# Read the Excel file
df = pd.read_excel(dataset_path)

# Explore the dataset
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.head())

# Print the column names and check for the presence of 'target'
print(df.columns)
print(df.tail())  # Corrected line
print(df.shape)
print(df['Class Attribute'].value_counts())

# Preprocessing
X = df.drop('Class Attribute', axis=1)
y = df['Class Attribute']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Define the ANN model function activation to avoid non-lineality
def create_ann_model(neurons=64, activation='relu', optimizer='adam'):
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(neurons,), activation=activation, solver=optimizer, max_iter=1000))
    return model

# Step 2: Create the MLPClassifier model
ann_model = create_ann_model()

# Step 3: Perform 10-fold cross-validation
cv_scores_ann = cross_val_score(ann_model, X_train, y_train, cv=10, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores (ANN):", cv_scores_ann)
print("Mean accuracy (ANN):", np.mean(cv_scores_ann))

# Step 4: Perform RandomizedSearchCV for hyperparameter tuning
param_dist_ann = {
    'mlpclassifier__hidden_layer_sizes': [(32,), (64,), (128,)],
    'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
    'mlpclassifier__solver': ['adam', 'sgd'],
}

random_search_ann = RandomizedSearchCV(ann_model, param_distributions=param_dist_ann, n_iter=10, cv=3, scoring='accuracy', random_state=42)
random_search_ann.fit(X_train, y_train)

# Print the best parameters and their corresponding accuracy
print("Best parameters (ANN):", random_search_ann.best_params_)
print("Best accuracy (ANN):", random_search_ann.best_score_)

# Step 5: Use the best model from RandomizedSearchCV
best_ann_model = random_search_ann.best_estimator_

# Step 6: Predict using the best ANN model
predict_X_train_ann = best_ann_model.predict(X_train)

# Step 7: Calculate accuracy and F1-score
accuracy_prediction_ann = accuracy_score(predict_X_train_ann, y_train)
f1_prediction_ann = f1_score(predict_X_train_ann, y_train, average='weighted')  # Change 'binary' to 'weighted'

print('Model accuracy (ANN):', accuracy_prediction_ann)
print('Model F1-score (ANN):', f1_prediction_ann)

# Step 8: Generate and print the confusion matrix
conf_matrix_ann = confusion_matrix(y_train, predict_X_train_ann)
print("Confusion Matrix (ANN):\n", conf_matrix_ann)

# Create a heatmap for the confusion matrix
sns.heatmap(conf_matrix_ann, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - ANN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 10: Evaluate the model on the test set
test_predictions_ann = best_ann_model.predict(X_test)

# Calculate accuracy and F1-score on the test set
accuracy_test_ann = accuracy_score(test_predictions_ann, y_test)
f1_test_ann = f1_score(test_predictions_ann, y_test, average='weighted')

print('Test Set Accuracy (ANN):', accuracy_test_ann)
print('Test Set F1-score (ANN):', f1_test_ann)

# Step 11: Generate and print the confusion matrix for the test set
conf_matrix_test_ann = confusion_matrix(y_test, test_predictions_ann)
print("Confusion Matrix (Test Set - ANN):\n", conf_matrix_test_ann)

# Step 12: Visualize the confusion matrix for the test set
sns.heatmap(conf_matrix_test_ann, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix (Test Set - ANN)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

