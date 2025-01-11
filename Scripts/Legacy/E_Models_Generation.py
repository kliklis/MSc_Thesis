# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, silhouette_score
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

def Log(message):
    if log_flag == False: return
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

def save_model(model, filename='_model.pkl'):   
    Log("Saving the model...")
    joblib.dump(model, filename)

    Log("Process completed. Model saved as 'adhd_random_forest_model.pkl'.")

def Menu():
    global methods

    print("Please select a machine learning algorithm:")
    for idx, method in methods.items():
        print(f"{idx + 1}. {method}")  # Display as 1-based indexing

    while True:
        try:
            # Input for user choice
            choice = int(input("Enter the number corresponding to your choice: "))
            if 1 <= choice <= len(methods):
                return choice - 1  # Convert to 0-based indexing
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(methods)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def export_dataframe(file_name, df):
    Log(f"Exporting dataframe to {file_name}...")
    df.to_csv(file_name, index=False)
    Log(f"Dataframe successfully exported to {file_name}.")

def kmeans_clustering(X):
    Log("Starting K-Means clustering...")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Evaluate clustering using silhouette score
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score: {sil_score:.2f}")

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title('K-Means Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # Return clusters
    return clusters

def handle_missing_values(dataset):
    if dataset.isna().any().any():
        Log("Handling missing values...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    else:
        dataset_imputed = dataset
    
    #dataset_imputed = dataset.dropna()
    return dataset_imputed

def normalize_numeric_columns(dataset):
    Log("Normalizing numeric columns...")
    numeric_columns = dataset.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

    Log(f"Numeric columns normalized: {list(numeric_columns)}")
    return dataset

def preprocess_dataset(df):
    df['escaped'] = df['escaped'].astype(int)
    has_adhd_column = df.pop('has_adhd').astype(int)
    
    for col in ['riddle_1', 'riddle_2', 'riddle_3', 'riddle_4', 'riddle_5']:
        df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0)
        df[f'{col}_count'] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df.drop(['riddle_1', 'riddle_2', 'riddle_3', 'riddle_4', 'riddle_5'], axis=1, inplace=True)
    
    for col in ['ommision_errors', 'commision_errors']:
        df[f'{col}_sum'] = df[col].apply(lambda x: sum(x) if isinstance(x, list) and len(x) > 0 else 0)
    df.drop(['ommision_errors', 'commision_errors'], axis=1, inplace=True)
    
    for col in ['distraction_1_timestamps', 'distraction_2_timestamps', 'distraction_3_timestamps']:
        df[[f'{col}_start', f'{col}_response', f'{col}_resolution']] = df[col].apply(
            lambda x: pd.Series([x[0], x[1], x[2]]) if isinstance(x, list) and len(x) == 3 else pd.Series([np.nan, np.nan, np.nan])
        )
        df[f'{col}_response_time'] = df[f'{col}_response'] - df[f'{col}_start']
        df[f'{col}_resolution_time'] = df[f'{col}_resolution'] - df[f'{col}_response']
    df.drop(['distraction_1_timestamps', 'distraction_2_timestamps', 'distraction_3_timestamps'], axis=1, inplace=True)
    
    df['has_adhd'] = has_adhd_column
    
    return df

def load_dataset(file_path='A_Labeled.csv'):
    Log("Loading dataset...")
    dataset = pd.read_csv(file_path)
    return dataset

def process_dataset(dataset, file_path='A_Labeled_Preprocessed.csv'):
    Log("Preprocessing dataset...")
    dataset = preprocess_dataset(dataset)
    dataset = handle_missing_values(dataset)
    dataset = normalize_numeric_columns(dataset)
    export_dataframe(file_path, dataset)
    return dataset

def split_dataset_label(dataset):
    X = dataset.drop(['has_adhd'], axis=1)  # Features
    y = dataset['has_adhd']  # Target
    return (X,y)
    
def boost_minority_class(X,y,desired_minority_ratio=0.3):
    Log("Applying SMOTE for class imbalance...")
    # Apply SMOTE to balance the class distribution in a ratio 70-30
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X, y = smote.fit_resample(X, y)
    return (X,y)
    
def split_train_test(X, y, test_ratio=0.2):
    return train_test_split(X, y, test_size=test_ratio, random_state=42)

def calculate_metrics_confidence_interval(model, X, y, n_iterations=1000, confidence=0.95):
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    n_size = len(X)

    for _ in range(n_iterations):
        # Resample dataset
        X_resample, y_resample = resample(X, y, n_samples=n_size, random_state=None)
        # Make predictions
        y_pred = model.predict(X_resample)
        # Calculate metrics
        metrics["accuracy"].append(accuracy_score(y_resample, y_pred))
        metrics["precision"].append(precision_score(y_resample, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_resample, y_pred, zero_division=0))
        metrics["f1_score"].append(f1_score(y_resample, y_pred, zero_division=0))

    # Calculate confidence intervals for each metric
    confidence_intervals = {}
    for metric, scores in metrics.items():
        lower_bound = np.percentile(scores, ((1 - confidence) / 2) * 100)
        upper_bound = np.percentile(scores, (confidence + (1 - confidence) / 2) * 100)
        confidence_intervals[metric] = (lower_bound, upper_bound)

    return confidence_intervals

def evaluate_predictions(y_test, y_pred):
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
    print("____________________\n")
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)
    print("____________________\n")


def evaluate_class_imbalance(y):
    Log("Calculating class imbalance metrics...")
    # Class Imbalance Metrics
    if smote_applied:
        class_counts = pd.Series(y).value_counts()
    else:
        class_counts = y.value_counts()

    class_imbalance_ratio = class_counts[1] / class_counts[0]
    print("Class Imbalance Ratio:", class_imbalance_ratio)

def select_model(algorithm_index=0):
    global is_labeled,normalization_applied,smote_applied,desired_minority_boost_ratio,features_number,examples_number
    
    algorithm_index = 0
    print("\n________________\nGlobal Parameters:")
    print(f"  Dataset Labeled: {is_labeled}")
    print(f"  Dataset Features Number: {features_number}")
    print(f"  Dataset Examples Number: {examples_number}")
    print(f"  Normalization Applied: {normalization_applied}")
    print(f"  SMOTE Applied: {smote_applied}")
    print(f"  SMOTE Minority Class Boost Ratio: {desired_minority_boost_ratio}")
    print("________________\n")
    
    model = None
    if algorithm_index == 0:  # Random Forest
        Log("Random Forest algorithm selected...")
        model = RandomForestClassifier(
            n_estimators=2000,  # Number of trees in the forest; higher improves stability but increases time.
            max_depth=15,  # Maximum depth of each tree; higher allows more splits but risks overfitting.
            min_samples_split=5,  # Minimum samples needed to split a node; higher reduces overfitting.
            min_samples_leaf=5,  # Minimum samples in a leaf; higher prevents small leaves.
            max_features=math.ceil(math.sqrt(features_number) * 3.5),  # Number of features considered per split; higher allows more complexity.
            bootstrap=False,  # Whether to use bootstrap sampling; False uses the full dataset for each tree.
            random_state=42  # Seed for reproducibility; ensures consistent results across runs.
        )
        
        print("\n________________\nHyperparameters:")
        print(f"  n_estimators: {model.n_estimators}")
        print(f"  max_depth: {model.max_depth}")
        print(f"  min_samples_split: {model.min_samples_split}")
        print(f"  min_samples_leaf: {model.min_samples_leaf}")
        print(f"  max_features: {model.max_features}")
        print(f"  bootstrap: {model.bootstrap}")
        print(f"  random_state: {model.random_state}")
        print("________________\n")
        
    elif algorithm_index == 1:  # Gradient Boosting
        Log("Gradient Boosting algorithm selected...")
        model = HistGradientBoostingClassifier(
        loss='log_loss',  # Loss function to optimize ('log_loss' for classification).
        learning_rate=0.05,  # Shrinks contribution of each tree; lower is slower but more stable.
        max_iter=1000,  # Number of boosting iterations; higher allows more complex models.
        max_leaf_nodes=31,  # Maximum leaf nodes per tree; higher captures more complexity.
        max_depth=8,  # Maximum tree depth; higher allows more splits but risks overfitting.
        min_samples_leaf=20,  # Minimum samples in a leaf; higher prevents small leaves.
        l2_regularization=0.0,  # Penalizes large weights; higher reduces overfitting risk.
        max_bins=255,  # Number of bins for continuous features; higher captures more detail.
        early_stopping='auto',  # Stops training early if validation performance stops improving.
        validation_fraction=0.01,  # Fraction of data for validation; higher leaves less for training.
        n_iter_no_change=100,  # Iterations without improvement before early stopping.
        scoring=None,  # Metric for early stopping; None uses the loss function.
        verbose=0,  # Controls output verbosity; 0 is silent, 1 shows progress.
        tol=1e-7,  # Tolerance for early stopping; lower requires stricter improvement.
        warm_start=False,  # Reuse solution of previous fit; useful for incremental learning.
        class_weight = {0.0: 0.4, 1.0: 0.6},#None,  # Class weights for imbalanced datasets; 'balanced' adjusts automatically.
        random_state=42  # Seed for reproducibility; ensures consistent splits across runs.
        )

        print("\n________________\nHyperparameters:")
        print(f"  loss: {model.loss}")
        print(f"  learning_rate: {model.learning_rate}")
        print(f"  max_iter: {model.max_iter}")
        print(f"  max_leaf_nodes: {model.max_leaf_nodes}")
        print(f"  max_depth: {model.max_depth}")
        print(f"  min_samples_leaf: {model.min_samples_leaf}")
        print(f"  l2_regularization: {model.l2_regularization}")
        print(f"  max_bins: {model.max_bins}")
        print(f"  early_stopping: {model.early_stopping}")
        print(f"  validation_fraction: {model.validation_fraction}")
        print(f"  n_iter_no_change: {model.n_iter_no_change}")
        print(f"  scoring: {model.scoring}")
        print(f"  verbose: {model.verbose}")
        print(f"  tol: {model.tol}")
        print(f"  categorical_features: {model.categorical_features}")
        print(f"  warm_start: {model.warm_start}")
        print(f"  class_weight: {model.class_weight}")
        print(f"  random_state: {model.random_state}")
        print("________________\n")
        
    elif algorithm_index == 2:  # k-Nearest Neighbors
        Log("k-Nearest Neighbors algorithm selected...")
        model = KNeighborsClassifier(
            n_neighbors=10,  # Number of neighbors to consider; higher smoothens decision boundaries.
            weights='distance',  # 'uniform': equal weight; 'distance': weight inversely proportional to distance.
            algorithm='auto',  # Algorithm to compute neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
            leaf_size=30,  # Leaf size for 'ball_tree' or 'kd_tree'; smaller values improve accuracy but increase time.
            p=1,  # Power parameter for the Minkowski distance metric; p=2 for Euclidean, p=1 for Manhattan.
            metric='minkowski',  # Distance metric to use; 'minkowski' is generalizable for p=1 or p=2. () #euclidean, manhattan, chebyshev, cosine
            metric_params=None,  # Additional parameters for the metric, if applicable.
            n_jobs=None  # Number of parallel jobs (-1 to use all CPUs); None for sequential computation.
        )

        # Print hyperparameters in the specified format
        print("\n________________\nHyperparameters:")
        print(f"  n_neighbors: {model.n_neighbors}")
        print(f"  weights: {model.weights}")
        print(f"  algorithm: {model.algorithm}")
        print(f"  leaf_size: {model.leaf_size}")
        print(f"  p: {model.p}")
        print(f"  metric: {model.metric}")
        print(f"  metric_params: {model.metric_params}")
        print(f"  n_jobs: {model.n_jobs}")
        print("________________\n")
        
    return model

def voting_predict(models, weights=None, soft_voting=False, example=None):

    if weights is None:
        weights = [1] * len(models)  # Equal weights if not provided

    if len(weights) != len(models):
        raise ValueError("The number of weights must match the number of models.")

    if example is None:
        raise ValueError("An example must be provided for prediction.")

    # Soft voting
    if soft_voting:
        probabilities = np.array([model.predict_proba([example])[0] * weight for model, weight in zip(models, weights)])
        avg_probabilities = np.sum(probabilities, axis=0) / np.sum(weights)
        predicted_value = np.argmax(avg_probabilities)

    # Hard voting
    else:
        predictions = np.array([model.predict([example])[0] for model in models])
        weighted_votes = np.zeros_like(np.unique(predictions), dtype=float)
        for pred, weight in zip(predictions, weights):
            weighted_votes[pred] += weight
        predicted_value = np.argmax(weighted_votes)

    return predicted_value

#MAIN    ____________________

# Correcting the data structure for supervised_methods and unsupervised_methods
supervised_methods = {
    0: 'Random Forest',
    1: 'Gradient Boost',
    2: 'K-Nearest Neighbors'
    # Neural Networks can be added later if needed
}

unsupervised_methods = {
    3: 'K-Means Clustering',
    4: 'DBSCAN'
}


log_flag = False
# Merge dictionaries into a single dictionary
methods = {**supervised_methods, **unsupervised_methods}

# Flag to track whether SMOTE was used
normalization_applied = True
smote_applied = True
desired_minority_boost_ratio = 0.3
is_labeled = True
features_number = 0
examples_number = 0

def main():
    global is_labeled,normalization_applied,smote_applied,desired_minority_boost_ratio,features_number,examples_number
    # ---------------- PARAMETERS
    Log("START")

    for iteration in range(3):
        algorithm_index = iteration#Menu()
        if algorithm_index >= 3: is_labeled = False

        # ---------------- DATASET READ
        #dataset = load_dataset(file_path = '../Datasets/A_Labeled.csv')
        dataset = load_dataset(file_path = '../Datasets/A_Labeled_Preprocessed.csv')
        features_number = len(dataset.columns)
        examples_number = dataset.shape[0]
        # ---------------- PREPROCESSING
        #dataset = process_dataset(dataset, file_path = '../Datasets/A_Labeled_Preprocessed.csv')
        dataset = handle_missing_values(dataset)
        
        if normalization_applied:
            dataset = normalize_numeric_columns(dataset)

        # ---------------- DATASET SPLIT
        if is_labeled:
            X,y = split_dataset_label(dataset)

        if is_labeled == True: 
            # ---------------- CLASS IMBALANCE
            if smote_applied == True:
                X,y = boost_minority_class(X,y,desired_minority_ratio=0.3)
            
            Log("Performing train-test split...")
            X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2)

            # ---------------- TRAINING
            model = select_model(iteration)
            model.fit(X_train, y_train)

            print(calculate_metrics_confidence_interval(model,  X_train, y_train, n_iterations=1000, confidence=0.95))

            '''
            models = [select_model(algorithm_index = 1),select_model(algorithm_index = 2),select_model(algorithm_index = 3)]
            
            for m in models:
                m.fit(X_train, y_train)
                
            voting_results = voting_predict(models)#, weights=None, soft_voting=True, example=None)
            '''
            
            Log("Making predictions...")
            # ---------------- PREDICTION
            y_pred = model.predict(X_test)

            # ---------------- EVALUATION
            evaluate_predictions(y_test, y_pred)
            evaluate_class_imbalance(y)

            # ---------------- SAVE MODEL
            save_model(model, filename = '../Models/' + methods[algorithm_index]+'.pkl')
            input()
        else:
            # Unlabeled dataset logic (K-Means Clustering)
            '''Log("Dataset is unlabeled. Applying K-Means clustering...")
            clusters = kmeans_clustering(X)
            dataset['cluster'] = clusters
            export_dataframe("kmeans_clustered_dataset.csv", dataset)
            Log("K-Means clustering completed and results saved.")'''
        
    Log("END")
    input()

if __name__ == "__main__":
    main()


#feature selection
#feature weights
#confidence interval
