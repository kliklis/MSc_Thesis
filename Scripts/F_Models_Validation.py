# Importing necessary libraries
import pandas as pd
import joblib
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import CustomUtils

def evaluate_model(dataset_path, models_dir):
    # Import dataset
    dataset = CustomUtils.import_dataset(file_path=dataset_path)
    if dataset is None:
        CustomUtils.Log("Dataset loading failed. Exiting...")
        return

    # Preprocess dataset if necessary
    #CustomUtils.Log("Handling missing values in the dataset...")
    #dataset = CustomUtils.handle_missing_values(dataset)

    # Split dataset into features and labels
    X, y = CustomUtils.split_dataset_label(dataset)

    # Define scoring metrics for cross-validation
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1_score': make_scorer(f1_score, zero_division=0)
    }

    # Iterate over supervised methods
    for idx, method_name in CustomUtils.supervised_methods.items():
        model_path = models_dir#f"{models_dir}/{method_name}.pkl"
        CustomUtils.Log(f"Loading model: {method_name} from {model_path}...")

        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            CustomUtils.Log(f"Model file {model_path} not found. Skipping...")
            continue

        CustomUtils.Log(f"Performing cross-validation for {method_name}...")
        results = {}

        # Perform cross-validation for each metric
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model, X, y, scoring=scorer, cv=5)
            results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }

        CustomUtils.Log(f"Cross-validation completed for {method_name}.")

        # Display results
        print(f"\nModel: {method_name}")
        for metric, values in results.items():
            print(f"{metric.capitalize()} - Mean: {values['mean']:.4f}, Std: {values['std']:.4f}")

# Example usage
def main():
    dataset_path = '../Datasets/A_Labeled_Preprocessed.csv'  # Path to your preprocessed dataset
    models_dir = '../Models'  # Directory where your models are saved
    
    CustomUtils.Log("Starting model evaluation...")
    evaluate_model(dataset_path, models_dir)
    CustomUtils.Log("Model evaluation completed.")

if __name__ == "__main__":
    main()
