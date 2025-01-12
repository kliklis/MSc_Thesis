import pandas as pd
import joblib
import CustomUtils

def label_unlabeled_dataset(model_path, unlabeled_dataset_path):
    """
    Labels an unlabeled dataset using a pre-trained model and exports it with a new column 'has_adhd'.

    Args:
        model_path (str): Path to the saved model file.
        unlabeled_dataset_path (str): Path to the unlabeled dataset (CSV file).

    Returns:
        pd.DataFrame: The labeled dataset with the 'has_adhd' column added.
    """
    import os

    # Log the process
    CustomUtils.Log(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        CustomUtils.Log("Model loaded successfully.")
    except FileNotFoundError:
        CustomUtils.Log(f"Error: Model file {model_path} not found.")
        return

    # Load the unlabeled dataset
    CustomUtils.Log(f"Loading unlabeled dataset from {unlabeled_dataset_path}...")
    dataset = CustomUtils.import_dataset(unlabeled_dataset_path)
    if dataset is None:
        CustomUtils.Log("Failed to load the unlabeled dataset. Exiting...")
        return

    # Perform predictions
    CustomUtils.Log("Predicting labels for the unlabeled dataset...")
    try:
        predictions = model.predict(dataset)
        dataset['has_adhd'] = predictions
        CustomUtils.Log("Labels predicted successfully.")
    except Exception as e:
        CustomUtils.Log(f"Error during prediction: {e}")
        return

    return dataset

def main():
    # Define paths
    model_path = '../Models/Random_Forest.pkl'  # Example path to a trained model
    unlabeled_dataset_path = '../Datasets/B_Unlabeled.csv'  # Path to the unlabeled dataset

    # Label the unlabeled dataset
    label_unlabeled_dataset(model_path, unlabeled_dataset_path)

if __name__ == "__main__":
    main()
