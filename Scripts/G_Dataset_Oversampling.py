import CustomUtils
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def autoencoder_dataset_oversampling(
    dataset,
    hidden_layer_size=None,
    latent_layer_size=None,
    num_samples=100,
    scaler=None,
    max_iter=1000,
    random_state=42
):
    """
    Augments a dataset using an autoencoder to generate synthetic samples.

    Args:
        dataset (np.ndarray): Input dataset as a NumPy array.
        hidden_layer_size (int): Number of neurons in the hidden layers. Default is half the feature size.
        latent_layer_size (int): Number of neurons in the bottleneck layer. Default is one-third the feature size.
        num_samples (int): Number of synthetic samples to generate.
        scaler: Scaler to normalize data. If None, uses StandardScaler.
        max_iter (int): Maximum number of iterations for the autoencoder training.
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Augmented dataset including original and synthetic samples.
    """
    # Ensure dataset is a NumPy array
    dataset = np.array(dataset)

    # Determine feature size
    n_features = dataset.shape[1]

    # Default sizes for hidden and latent layers if not provided
    if hidden_layer_size is None:
        hidden_layer_size = n_features // 2
    if latent_layer_size is None:
        latent_layer_size = n_features // 3

    # Scale the dataset
    if scaler is None:
        scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    # Define the autoencoder
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_size, latent_layer_size, hidden_layer_size),
        max_iter=max_iter,
        random_state=random_state
    )

    # Train the autoencoder
    autoencoder.fit(scaled_data, scaled_data)

    # Get latent representations
    latent_data = autoencoder.predict(scaled_data)

    # Generate synthetic samples
    synthetic_samples = []
    for _ in range(num_samples):
        # Add slight noise in the latent space
        perturbed_latent = latent_data + np.random.normal(0, 0.05, latent_data.shape)
        # Decode back to data space
        synthetic_sample = autoencoder.predict(perturbed_latent)
        synthetic_samples.append(synthetic_sample)

    synthetic_samples = np.vstack(synthetic_samples)

    # Scale back the synthetic samples to the original space
    synthetic_samples = scaler.inverse_transform(synthetic_samples)

    # Combine original and synthetic datasets
    augmented_dataset = np.vstack([dataset, synthetic_samples])

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
