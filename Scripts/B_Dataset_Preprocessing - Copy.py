import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import datetime
import CustomUtils

def extract_stats(dataset, column):
    dataset[f'{column}_mean'] = dataset[column].dropna().apply(lambda x: np.mean(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_std'] = dataset[column].dropna().apply(lambda x: np.std(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_min'] = dataset[column].dropna().apply(lambda x: np.min(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_max'] = dataset[column].dropna().apply(lambda x: np.max(list(map(float, x.split(',')))) if isinstance(x, str) else np.nan)
    dataset[f'{column}_count'] = dataset[column].dropna().apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    return dataset

def process_dataset(dataset):
    CustomUtils.Log("Preprocessing dataset...")

    # Define the original dataset column order
    original_columns = dataset.columns.tolist()

    # Select specific features to keep
    features_to_keep = [
        "clicks", "completion_progress", "session_duration",
        "input_forward", "input_backward", "input_left", "input_right",
        "riddle_1", "riddle_2", "riddle_3", "riddle_4", "riddle_5",
        "escaped", "has_adhd"
    ]

    df = pd.DataFrame()
    for feature in features_to_keep:
        if feature in dataset.columns:
            df[feature] = dataset[feature]

    # Extract stats for riddle columns
    for riddle in ['riddle_1', 'riddle_2', 'riddle_3', 'riddle_4', 'riddle_5']:
        if riddle in dataset.columns:
            df = extract_stats(df, riddle)

    # Drop original riddle columns after extracting stats
    df = df.drop(columns=['riddle_1', 'riddle_2', 'riddle_3', 'riddle_4', 'riddle_5'], errors='ignore')

    # Manually splitting `ommision_errors` into six columns
    omission_cols = ["ommision_1", "ommision_2", "ommision_3", "ommision_4", "ommision_5", "ommision_6"]
    df[omission_cols] = dataset["ommision_errors"].str.split(",", expand=True).astype(float)

    # Manually splitting `commision_errors` into six columns
    commission_cols = ["commision_1", "commision_2", "commision_3"]
    df[commission_cols] = dataset["commision_errors"].str.split(",", expand=True).astype(float)

    # Manually splitting `distraction_1_timestamps` into three columns
    distraction_1_cols = ["distraction_1_1", "distraction_1_2", "distraction_1_3"]
    df[distraction_1_cols] = dataset["distraction_1_timestamps"].str.split(",", expand=True).astype(float)

    # Manually splitting `distraction_2_timestamps` into three columns
    distraction_2_cols = ["distraction_2_1", "distraction_2_2", "distraction_2_3"]
    df[distraction_2_cols] = dataset["distraction_2_timestamps"].str.split(",", expand=True).astype(float)

    # Manually splitting `distraction_3_timestamps` into three columns
    distraction_3_cols = ["distraction_3_1", "distraction_3_2", "distraction_3_3"]
    df[distraction_3_cols] = dataset["distraction_3_timestamps"].str.split(",", expand=True).astype(float)

    # Reorder columns to match the original order
    final_columns = [col for col in original_columns if col in df.columns] + [col for col in df.columns if col not in original_columns]
    df = df[final_columns]
    
    # Handle missing values
    df = CustomUtils.handle_missing_values(df)

    return df

def main():
    # Load dataset
    dataset = CustomUtils.import_dataset(file_path='../Datasets/A_Labeled.csv')
    if dataset is None:
        CustomUtils.Log("Terminating program due to missing dataset.")
        return

    # Process dataset
    processed_dataset = process_dataset(dataset)

    # Export processed dataset
    CustomUtils.export_dataset('../Datasets/A_Labeled_Preprocessed.csv', processed_dataset)

if __name__ == "__main__":
    main()
