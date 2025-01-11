import pandas as pd
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def load_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    return df.drop(columns=['has_adhd'], errors='ignore')

def predict_labels(model, data):
    return model.predict(data)

def save_predictions(df, predictions, output_path):
    df['has_adhd'] = predictions
    df.to_csv(output_path, index=False)

def main():
    model_path = '../Models/Random Forest.pkl'
    input_csv = '../Datasets/Ba_Unlabeled.csv'#'A_Labeled_Preprocessed.csv'
    output_csv = '../Datasets/'+input_csv[:-3]+'_Predicted.csv'

    model = load_model(model_path)
    dataset = load_dataset(input_csv)
    features = preprocess_data(dataset)
    predictions = predict_labels(model, features)
    save_predictions(dataset, predictions, output_csv)

if __name__ == "__main__":
    main()
