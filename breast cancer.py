# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load and prepare dataset (Breast Cancer dataset)
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data[:, :2]  # Use only first two features for 2D visualization
y = breast_cancer.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Function to plot decision boundaries for 2D data
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Step 2: Train SVM with linear kernel
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

# Step 3: Visualize decision boundary (linear kernel)
plot_decision_boundary(svm_linear, X_train, y_train, 'SVM with Linear Kernel')

# Step 2: Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1)
svm_rbf.fit(X_train, y_train)

# Step 3: Visualize decision boundary (RBF kernel)
plot_decision_boundary(svm_rbf, X_train, y_train, 'SVM with RBF Kernel')

# Step 4: Hyperparameter tuning for RBF kernel using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters from GridSearch: {grid_search.best_params_}")

# Train model with best parameters
svm_best = grid_search.best_estimator_

# Step 5: Evaluate performance with cross-validation
cv_scores = cross_val_score(svm_best, X_scaled, y, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean()}")

# Evaluate on test set
y_pred = svm_best.predict(X_test)
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))
print(f"Test set accuracy: {accuracy_score(y_test, y_pred)}")

# Visualize decision boundary with best model
plot_decision_boundary(svm_best, X_train, y_train, 'SVM with RBF Kernel (Tuned)')
