# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime

# Logging function to print timestamped messages
def Log(message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

def Menu():
    methods = [
        'Random Forest', 
        'Gradient Boost', 
        'K-Nearest Neighbors', 
        'Neural Networks', 
        'K-Means Clustering', 
        'DBSCAN'
    ]
    
    print("Please select a machine learning algorithm:")
    for idx, method in enumerate(methods, start=1):
        print(f"{idx}. {method}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 1 <= choice <= len(methods):
                return choice - 1  # return the index (0-based)
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(methods)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    Log("START")

    answer_index = Menu()

    Log("Loading dataset...")
    # Load dataset
    dataset = pd.read_csv('dataset.csv')

    # Splitting features and target
    X = dataset.drop(['user_id', 'has_adhd'], axis=1)  # Features
    y = dataset['has_adhd']  # Target

    # Flag to track whether SMOTE was used
    smote_applied = False

    if False:
        Log("Applying SMOTE for class imbalance...")
        # Apply SMOTE to balance the class distribution
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        smote_applied = True

        # Train-test split on the balanced data
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    else:
        Log("Performing train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Log("Training Random Forest model...")
    # Training Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    Log("Making predictions...")
    # Making predictions
    y_pred = rf_model.predict(X_test)

    Log("Calculating evaluation metrics...")
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)

    Log("Calculating class imbalance metrics...")
    # Class Imbalance Metrics
    if smote_applied:
        class_counts = y_resampled.value_counts()
    else:
        class_counts = y.value_counts()

    class_imbalance_ratio = class_counts[1] / class_counts[0]
    print("Class Imbalance Ratio:", class_imbalance_ratio)

    Log("Saving the Random Forest model...")
    # Saving the model
    joblib.dump(rf_model, 'adhd_random_forest_model.pkl')

    Log("Process completed. Model saved as 'adhd_random_forest_model.pkl'.")

if __name__ == "__main__":
    main()
