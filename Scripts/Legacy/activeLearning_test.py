import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import modAL
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
X, y = make_classification(
    n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42
)

# Step 2: Split data into initial training and unlabeled pools
X_initial, X_pool, y_initial, y_pool = train_test_split(X, y, test_size=0.95, random_state=42)

# Step 3: Initialize the ActiveLearner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_initial,
    y_training=y_initial
)

# Step 4: Active learning loop
n_queries = 20
performance_history = []

for idx in range(n_queries):
    # Query the most uncertain sample
    query_idx, query_instance = learner.query(X_pool)
    
    # Simulate labeling by the oracle
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, ))
    
    # Remove the queried instance from the unlabeled pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    
    # Record the learner's performance
    performance_history.append(learner.score(X, y))

# Step 5: Plot the performance over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_queries + 1), performance_history, marker='o')
plt.title('Model Accuracy Over Queries')
plt.xlabel('Query Number')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
