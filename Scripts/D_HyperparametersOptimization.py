import optuna
import CustomUtils
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
import pandas as pd



# Merge dictionaries into a single dictionary
CustomUtils.methods

# Define the objective function for Optuna
def objective(trial, method = CustomUtils.methods[0]):
    dataset = CustomUtils.import_dataset(file_path = '../Datasets/A_Labeled_Preprocessed.csv')
    #dataset = dataset.drop(columns = ['ommision_5', 'distraction_1_2', 'riddle_2_count', 'riddle_3_count', 'riddle_4_count', 'riddle_3_std', 'riddle_5_std', 'riddle_2_min', 'riddle_2_max', 'riddle_3_mean', 'clicks', 'riddle_4_min', 'riddle_3_min', 'riddle_1_count', 'completion_progress'])
    X_train, X_test, y_train, y_test = CustomUtils.custom_train_test_split(dataset, test_set_size=0.3, current_random_state=42)
    # Select a classifier
    #classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "GradientBoosting", "KNN"])

    if method == CustomUtils.methods[0]:
        # Hyperparameter tuning for Random Forest
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_int("max_features", 1, 25)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        random_state = 42

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )

    elif method == CustomUtils.methods[1]:
        # Hyperparameter tuning for Gradient Boosting
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 50)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)
        max_bins = trial.suggest_int("max_bins", 50, 255)
        validation_fraction = trial.suggest_float("validation_fraction", 0.01, 0.3)
        n_iter_no_change = trial.suggest_int("n_iter_no_change", 10, 100)
        tol = trial.suggest_float("tol", 1e-7, 1e-3, log=True)

        model = HistGradientBoostingClassifier(
            loss='log_loss',
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            early_stopping='auto',
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            class_weight={0.0: 0.4, 1.0: 0.6},
            random_state=42
        )

    elif method == CustomUtils.methods[2]:
        # Hyperparameter tuning for K-Nearest Neighbors
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        leaf_size = trial.suggest_int("leaf_size", 10, 50)
        p = trial.suggest_int("p", 1, 2)
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski", "chebyshev", "cosine"])

        if metric == "cosine" and algorithm != "brute":
            algorithm = "brute"  # Ensure "cosine" works with the correct algorithm

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=None,  # Additional parameters for the metric, if applicable
            n_jobs=None  # Number of parallel jobs (-1 to use all CPUs)
        )

    elif method == CustomUtils.methods[3]:
        # Hyperparameter tuning for K-Means
        n_clusters = trial.suggest_int("n_clusters", 2, 20)
        init = trial.suggest_categorical("init", ["k-means++", "random"])
        n_init = trial.suggest_int("n_init", 10, 50)
        max_iter = trial.suggest_int("max_iter", 100, 500)
        tol = trial.suggest_float("tol", 1e-4, 1e-2, log=True)

        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )

    elif method == CustomUtils.methods[4]:
        # Hyperparameter tuning for DBSCAN
        eps = trial.suggest_float("eps", 0.1, 1.0)
        min_samples = trial.suggest_int("min_samples", 2, 10)
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )

    # Perform cross-validation and return the mean accuracy
    #"accuracy" "precision" "recall" "f1" "roc_auc" "log_loss"
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro']
    cv_results = cross_validate(model, X_train, y_train, cv=3, scoring=scoring_metrics)
    
    # Compute mean of each metric
    accuracy = cv_results['test_accuracy'].mean()
    precision = cv_results['test_precision_macro'].mean()
    recall = cv_results['test_recall_macro'].mean()
    score = (0.30 * accuracy) + (0.20 * precision) + (0.5 * recall)
    print(f"\nAccuracy: {accuracy}, \nPrecision: {precision}, \nRecall: {recall}, \nWeightedScore: {score}")
    
    return score

def optimize_hyperparameters(method_to_be_optimized=CustomUtils.methods[0], trials = 100):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, method=method_to_be_optimized), n_trials=trials)

    # Print the best hyperparameters
    print("\nBest trial:")
    print(f"  Value: {study.best_trial.value}")
    print("\n  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"\n    {key}: {value}")
    return study.best_trial.params

def main():
    optimize_hyperparameters()

if __name__ == "__main__":
    main()