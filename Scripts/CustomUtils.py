from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml

log_flag = True

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
# Merge dictionaries into a single dictionary
methods = {**supervised_methods, **unsupervised_methods}

# Logging function to print timestamped messages
def Log(message):
    if not log_flag:
        return
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")

def log_break(message='\n', pause=False):
    if pause:
        input(message)
    else:
        print(message)

def import_dataset(file_path='A_Labeled.csv'):
    Log("Importing dataset...")
    try:
        dataset = pd.read_csv(file_path)
        Log("Dataset imported successfully.")
        return dataset
    except FileNotFoundError:
        Log(f"Error: File {file_path} not found.")
        return None

def export_dataset(file_name, df):
    Log(f"Exporting dataset to {file_name}...")
    try:
        df.to_csv(file_name, index=False)
        Log(f"Dataset successfully exported to {file_name}.")
    except Exception as e:
        Log(f"Error exporting dataset: {e}")

def save_model(model, filename='_model.pkl'):   
    Log("Saving the model...")
    joblib.dump(model, filename)

    Log("Process completed. Model saved as " + filename + ".")

def handle_missing_values(dataset):
    # Drop columns with all missing values
    missing_cols = dataset.columns[dataset.isna().all()]
    dataset = dataset.dropna(axis=1, how="all")
    print(f"Dropped columns with all missing values: {list(missing_cols)}")
    
    # Drop columns with constant values
    #constant_cols = dataset.columns[dataset.nunique() <= 1]
    #dataset = dataset.drop(columns=constant_cols)
    #print(f"Dropped constant columns: {list(constant_cols)}")
    
    # Impute missing values for remaining columns
    imputer = SimpleImputer(strategy="mean")
    numeric_data = dataset.select_dtypes(include=['float64', 'int64'])
    imputed_data = imputer.fit_transform(numeric_data)
    numeric_imputed = pd.DataFrame(imputed_data, columns=numeric_data.columns)
    
    # Combine numeric and non-numeric columns
    non_numeric_data = dataset.select_dtypes(exclude=['float64', 'int64'])
    dataset_imputed = pd.concat([numeric_imputed, non_numeric_data], axis=1)
    
    return dataset_imputed

def handle_missing_values_2(dataset):
    if dataset.isna().any().any():
        Log("Handling missing values...")
        imputer = SimpleImputer(strategy='mean')
        dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    else:
        dataset_imputed = dataset
    
    #dataset_imputed = dataset.dropna()
    return dataset_imputed

def split_dataset_label(dataset):
    X = dataset.drop(['has_adhd'], axis=1)  # Features
    y = dataset['has_adhd']  # Target
    return (X,y)

def custom_train_test_split(dataset, test_set_size=0.3, current_random_state=42):
    X,y = split_dataset_label(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=current_random_state)
    return (X_train, X_test, y_train, y_test)

def boost_minority_class(dataset,desired_minority_ratio=0.3):
    Log("Applying SMOTE for class imbalance...")
    X,y = split_dataset_label(dataset)
    # Apply SMOTE to balance the class distribution in a ratio 70-30
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X, y = smote.fit_resample(X, y)
    dataset = pd.concat([X, y], axis=1)
    return (dataset)

def get_dataset_info(dataset):
    num_features = dataset.shape[1]
    dataset_size = dataset.shape[0]
    Log(f"Dataset info: {num_features} features, {dataset_size} rows.")
    return (num_features, dataset_size)

def drop_features(dataset, features_to_drop):
    Log(f"Attempting to drop features: {features_to_drop}")
    try:
        dataset_dropped = dataset.drop(columns=features_to_drop, errors='ignore')
        Log(f"Successfully dropped features: {features_to_drop}")
        return dataset_dropped
    except Exception as e:
        Log(f"Error dropping features: {e}")
        return dataset

def keep_features(dataset, features_to_keep):
    Log(f"Attempting to keep only features: {features_to_keep}")
    try:
        dataset_kept = dataset[features_to_keep]
        Log(f"Successfully kept only features: {features_to_keep}")
        return dataset_kept
    except KeyError as e:
        Log(f"Error keeping features: {e}. Some features may not exist in the dataset.")
        return dataset
    except Exception as e:
        Log(f"Unexpected error: {e}")
        return dataset
    
def dict_to_yaml(data, file_path):
    Log(f"Exporting dictionary to YAML file at {file_path}...")
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
        Log(f"YAML file '{file_path}' created successfully.")
    except Exception as e:
        Log(f"Failed to export dictionary to YAML: {e}")

def yaml_to_dict(file_path):
    Log(f"Reading dictionary from YAML file at {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        Log(f"YAML file '{file_path}' read successfully.")
        return data
    except FileNotFoundError:
        Log(f"Error: File {file_path} not found.")
    except yaml.YAMLError as e:
        Log(f"Error reading YAML file: {e}")
    except Exception as e:
        Log(f"Unexpected error: {e}")
    return None

def add_noise(file_path, noise_level=0.05):
    Log("Adding noise to the dataset...")
    try:
        dataset = import_dataset(file_path)
        numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_columns:
            # Determine if column contains single or multiple values
            if dataset[col].dtype == 'int64':
                noise = (np.random.normal(loc=0, scale=noise_level * dataset[col].std(), size=dataset[col].shape)).astype(int)
                dataset[col] += noise
            elif dataset[col].dtype == 'float64':
                noise = np.random.normal(loc=0, scale=noise_level * dataset[col].std(), size=dataset[col].shape)
                dataset[col] += noise
        
        Log("Noise added successfully.")
        return dataset
    except Exception as e:
        Log(f"Error adding noise: {e}")
        return None

