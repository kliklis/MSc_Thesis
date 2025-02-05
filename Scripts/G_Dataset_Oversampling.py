import CustomUtils
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def unlabeled_dataset_oversampling(dataset_path, size_increment_percentage):
    """
    Augments an unlabeled dataset with synthetic samples based on a size increment percentage.

    Args:
        dataset_path (str): Path to the unlabeled dataset (CSV format).
        size_increment_percentage (float): Desired percentage increase in dataset size (e.g., 50 for 50%).

    Returns:
        pd.DataFrame: Augmented dataset including synthetic samples.
    """
    from sklearn.cluster import KMeans
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    # Import dataset
    dataset = CustomUtils.import_dataset(file_path=dataset_path)

    if dataset is None or dataset.empty:
        raise ValueError("The dataset is None or empty. Cannot perform augmentation.")

    CustomUtils.Log(f"Original dataset size: {dataset.shape[0]} rows.")

    # Assign pseudo-labels using KMeans clustering
    num_clusters = max(2, len(dataset) // 50)  # Choose clusters dynamically
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    pseudo_labels = kmeans.fit_predict(dataset)

    # Count samples per pseudo-class
    class_counts = Counter(pseudo_labels)
    min_samples_per_class = min(class_counts.values())
    
    # Determine the appropriate k_neighbors for SMOTE
    k_neighbors = max(1, min(min_samples_per_class - 1, 5))  # Ensure k_neighbors <= samples per class

    # Determine desired number of samples
    num_samples = int(len(dataset) * (size_increment_percentage / 100))

    CustomUtils.Log(f"Applying SMOTE for {num_samples} synthetic samples with k_neighbors={k_neighbors}...")
    smote = SMOTE(sampling_strategy=(num_samples / len(dataset)), k_neighbors=k_neighbors, random_state=42)
    synthetic_data, _ = smote.fit_resample(dataset, pseudo_labels)

    # Create DataFrame for synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=dataset.columns)

    # Combine original and synthetic datasets
    augmented_dataset = pd.concat([dataset, synthetic_df[len(dataset):]], ignore_index=True)

    CustomUtils.Log(f"Augmented dataset size: {augmented_dataset.shape[0]} rows.")

    return augmented_dataset

def autoencoder_dataset_oversampling(
    dataset_path,
    hidden_layer_size=None,
    latent_layer_size=None,
    num_samples=100,
    scaler=None,
    max_iter=1000,
    random_state=42
):
    """
    Augments a dataset from a CSV file using an autoencoder to generate synthetic samples.

    Args:
        dataset_path (str): Path to the input CSV file.
        hidden_layer_size (int): Number of neurons in the hidden layers. Default is half the feature size.
        latent_layer_size (int): Number of neurons in the bottleneck layer. Default is one-third the feature size.
        num_samples (int): Number of synthetic samples to generate.
        scaler: Scaler to normalize data. If None, uses StandardScaler.
        max_iter (int): Maximum number of iterations for the autoencoder training.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Augmented dataset including original and synthetic samples.
    """
    # Load the dataset
    dataset = CustomUtils.import_dataset(file_path=dataset_path)
    
    if num_samples==0: return dataset

    if dataset is None or dataset.empty:
        raise ValueError("The dataset is None or empty. Cannot perform oversampling.")

    # Determine the number of features
    n_features = dataset.shape[1]
    if n_features == 0:
        raise ValueError("The dataset must have at least one feature.")

    # Default sizes for hidden and latent layers if not provided
    hidden_layer_size = hidden_layer_size or max(1, n_features // 2)
    latent_layer_size = latent_layer_size or max(1, n_features // 3)

    # Scale the dataset
    scaler = scaler or StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    # Define the autoencoder
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_size, latent_layer_size, hidden_layer_size),
        max_iter=max_iter,
        random_state=random_state
    )

    # Train the autoencoder
    autoencoder.fit(scaled_data, scaled_data)

    # Generate synthetic samples
    synthetic_samples = []
    latent_data = autoencoder.predict(scaled_data)
    for _ in range(num_samples):
        perturbed_latent = latent_data + np.random.normal(0, 0.05, latent_data.shape)
        synthetic_sample = autoencoder.predict(perturbed_latent)
        synthetic_samples.append(synthetic_sample)

    synthetic_samples = np.vstack(synthetic_samples)
    synthetic_samples = scaler.inverse_transform(synthetic_samples)

    # Create a DataFrame for synthetic samples
    synthetic_df = pd.DataFrame(synthetic_samples, columns=dataset.columns)

    # Combine original and synthetic datasets
    augmented_dataset = pd.concat([dataset, synthetic_df], ignore_index=True)

    return augmented_dataset


# Example usage:
# dataset = np.random.rand(20, 5)  # Small dataset with 5 features
# augmented_data = autoencoder_dataset_oversampling(dataset, num_samples=50)



def dataset_oversampling(filepath=""):
    dataset = CustomUtils.import_dataset(file_path=filepath)
    CustomUtils.boost_minority_class(dataset)
    return dataset

def main():
    pass

if __name__ == "__main__":
    main()
