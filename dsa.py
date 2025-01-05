# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

#  Generate synthetic data
# Generate a synthetic dataset for illustration
n_samples = 10000  # Total number of samples
n_features = 6  # Including target 'IsFraud'

# Simulate dataset columns
data = {
    'Amount': np.random.lognormal(mean=2.0, sigma=1.0, size=n_samples),  # Transaction amount
    'Location': np.random.choice([1, 2, 3], size=n_samples),  # 1: Local, 2: Different city, 3: International
    'Time': np.random.choice(range(0, 24), size=n_samples),  # Hour of the transaction (24-hour format)
    'CardDetails': np.random.choice([0, 1], size=n_samples),  # 1: Active, 0: Suspended
    'Age': np.random.randint(18, 70, size=n_samples),  # Age of the user
    'TransactionFrequency': np.random.choice([1, 2, 3, 4], size=n_samples),  # Frequency of transactions
    'IsFraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])  # Fraud (0: Non-Fraud, 1: Fraudulent)
}

# Create DataFrame from synthetic data
df = pd.DataFrame(data)

#Data Preprocessing
print("Initial Data Overview:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

#  Feature Engineering
# We can extract more features like 'HourOfDay' from 'Time' if needed, but let's keep it simple for now.

# Encoding categorical variables
# Encoding 'Location' and 'CardDetails' as these are categorical features.
le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])
df['CardDetails'] = le.fit_transform(df['CardDetails'])

#  Splitting Data into Features and Target
X = df.drop(columns=['IsFraud'])
y = df['IsFraud']

#  Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Handling Class Imbalance
# As fraud transactions are often much fewer than non-fraud transactions, we need to balance the dataset.
# We will use SMOTE (Synthetic Minority Oversampling Technique) to oversample the minority class.
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

#  Model Selection and Training
# We will try multiple models for comparison.

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_balanced, y_train_balanced)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Support Vector Classifier (SVC)
svc_model = SVC(random_state=42)
svc_model.fit(X_train_balanced, y_train_balanced)

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_balanced, y_train_balanced)

#  Model Evaluation

# Predicting with Decision Tree
dt_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Predicting with Random Forest
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Predicting with SVC
svc_pred = svc_model.predict(X_test_scaled)
svc_accuracy = accuracy_score(y_test, svc_pred)

# Predicting with Logistic Regression
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Predicting with KNN
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Step 9: Comparing Model Performance

print("\nModel Accuracy Scores:")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVC Accuracy: {svc_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"KNN Accuracy: {knn_accuracy:.4f}")

# Confusion Matrices for each model
models = ['Decision Tree', 'Random Forest', 'SVC', 'Logistic Regression', 'KNN']
predictions = [dt_pred, rf_pred, svc_pred, lr_pred, knn_pred]

for i, model in enumerate(models):
    print(f"\nConfusion Matrix for {model}:")
    cm = confusion_matrix(y_test, predictions[i])
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f"Confusion Matrix: {model}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Classification Report
# We can generate classification reports to evaluate precision, recall, f1-score, and support for each class.

for i, model in enumerate(models):
    print(f"\nClassification Report for {model}:")
    print(classification_report(y_test, predictions[i]))

#  Hyperparameter Tuning with GridSearchCV
# Hyperparameter tuning for Random Forest Classifier as an example
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train_balanced, y_train_balanced)

print("\nBest Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)

#  Feature Importance from Random Forest
# Random Forest can provide feature importances, which helps in understanding which features are most important.
rf_feature_importances = rf_model.feature_importances_
features = X.columns

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=rf_feature_importances)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

#  Dimensionality Reduction (Optional)
# Applying PCA to reduce the feature space to 2D for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train_balanced, palette='viridis')
plt.title('PCA of Training Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
