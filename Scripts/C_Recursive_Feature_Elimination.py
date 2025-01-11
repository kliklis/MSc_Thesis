import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import CustomUtils

def perform_rfe_unranked(dataset, target_column, n_features_to_select=10):
    CustomUtils.Log("Performing Recursive Feature Elimination (RFE)...")
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Adjust max_iter if needed

    # Perform RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_scaled, y)

    selected_features = X.columns[rfe.support_]
    CustomUtils.Log(f"Selected Features: {list(selected_features)}")
    return selected_features

def perform_rfe_ranked(dataset, target_column, n_features_to_select=10):
    CustomUtils.Log("Performing Recursive Feature Elimination (RFE)...")
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)  # Adjust max_iter if needed

    # Perform RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_scaled, y)

    # Get feature rankings
    rankings = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_
    }).sort_values(by='Ranking')

    selected_features = rankings[rankings['Ranking'] == 1]['Feature'].tolist()
    CustomUtils.Log(f"Selected Features (sorted by significance): {selected_features}")
    return selected_features

def main():
    dataset = CustomUtils.import_dataset(file_path='../Datasets/A_Labeled_Preprocessed.csv')
    if dataset is None:
        CustomUtils.Log("Terminating program due to missing dataset.")
        return

    # Perform RFE
    target_column = 'has_adhd'  # Specify your target column
    selected_features = perform_rfe_ranked(dataset, target_column, n_features_to_select=10)

    CustomUtils.Log(f"Top features selected: {selected_features}")

    print(list(selected_features['Feature'].tail(15)))

if __name__ == "__main__":
    main()
