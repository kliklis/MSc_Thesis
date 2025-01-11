import A_Dataset_Generation
import B_Dataset_ML_Friendly_Convertion
import CustomUtils
import C_Recursive_Feature_Elimination
import D_HyperparametersOptimization
import E_Models_Generation
import F_Models_Validation
import G_Dataset_Oversampling
import H_Labels_Inferencing
import CustomUtils

'''

Datasets:
A_Labeled: Generated.
B_Unlabeled: Given.
B_Clustered: Given.
B_SemiSupervised: Predicted.
B_ActiveLearning: Predicted.
C_Labeled: Pending...

Outline:
A_Dataset_Generation: Generate raw datasets for analysis.
B_Dataset_ML_Friendly_Convertion: Preprocess datasets to make them suitable for ML (e.g., cleaning, encoding).
C_Recursive_Feature_Elimination: Perform feature selection to identify the most relevant features.
D_HyperparametersOptimization: Optimize model hyperparameters for best performance.
E_Models_Generation: Train machine learning models with selected features and optimized hyperparameters.
F_Models_Validation: Validate trained models using metrics and cross-validation.
G_Labels_Prediction_Ba: Use trained models to predict labels on new or unseen data.
CustomUtils: Provide utility functions for logging, dataset handling, and reusable tasks.

Workflow:
W1: A_Dataset_Generation -> A -> B_Dataset_ML_Friendly_Convertion -> C_Recursive_Feature_Elimination -> D_HyperparametersOptimization -> E_Models_Generation -> F_Models_Validation
W2: Ba -> B_Dataset_ML_Friendly_Convertion -> D_HyperparametersOptimization -> E_Models_Generation -> F_Models_Validation
W3: Bb -> B_Dataset_ML_Friendly_Convertion -> C_Recursive_Feature_Elimination -> D_HyperparametersOptimization -> E_Models_Generation (Active Learning) -> F_Models_Validation
W4: Ba -> G_Labels_Prediction_Ba (Semi-Supervised) -> Ba_PredictedLabels

Results:
C_Recursive_Feature_Elimination: exports .yaml with the selected features of the dataset.
D_HyperparametersOptimization: exports .yaml file with the optimum hyperparameters.
E_Models_Generation: imports .yaml file with the optimum hyperparameters.
D_HyperparametersOptimization: exports .yaml file with the optimum hyperparameters.
E_Models_Generation: exports the models in .pkl file.
F_Models_Validation: exports a .csv file with the validation results.
G_Labels_Prediction_Ba: exports Ba_PredictedLabels

'''

def main():

    A_Labeled_path="../Datasets/A_Labeled.csv"
    A_Labeled_Preprocessed_path="../Datasets/A_Labeled_Preprocessed.csv"
    A_Labeled_Preprocessed_Selected_Features_path="../Datasets/A_Labeled_Preprocessed_Selected_Features.csv"
    B_Unlabeled_path="../Datasets/B_Unlabeled.csv"
    B_Unlabeled_Oversampled_path="../Datasets/B_Unlabeled_Oversampled.csv"
    B_Clustered_path="../Datasets/B_Clustered.csv"
    B_SemiSupervised_path="../Datasets/B_SemiSupervised.csv"
    B_ActiveLearning_path="../Datasets/B_ActiveLearning.csv"
    C_Labeled_path="../Datasets/C_Labeled.csv"

    selected_workflow = 'W1'
    

    if selected_workflow == 'W1':

        ### A
        #Generate A_Labeled.csv Dataset
        A_Labeled_Dataset = A_Dataset_Generation.generate_dataset(
        num_players=1000,
        adhd_ratio=0.15,
        room_difficulty=0.6)
        dataset_info=CustomUtils.get_dataset_info(A_Labeled_Dataset)
        CustomUtils.export_dataset(A_Labeled_path,A_Labeled_Dataset )

        ### B
        A_Labeled_Dataset = CustomUtils.import_dataset(file_path=A_Labeled_path)
        A_Labeled_Dataset_Preprocessed = B_Dataset_ML_Friendly_Convertion.process_dataset(A_Labeled_Dataset)
        dataset_info=CustomUtils.get_dataset_info(A_Labeled_Dataset_Preprocessed)
        CustomUtils.export_dataset(A_Labeled_Preprocessed_path, A_Labeled_Dataset_Preprocessed )

        ### C
        A_Labeled_Dataset_Preprocessed = CustomUtils.import_dataset(file_path=A_Labeled_Preprocessed_path)
        selected_features = C_Recursive_Feature_Elimination.perform_rfe_ranked(A_Labeled_Dataset_Preprocessed, 'has_adhd', n_features_to_select=round(dataset_info[0]*0.75))
        selected_features.append('has_adhd')
        A_Labeled_Dataset_Preprocessed_Selected_Features = CustomUtils.keep_features(A_Labeled_Dataset_Preprocessed, selected_features)
        dataset_info=CustomUtils.get_dataset_info(A_Labeled_Dataset_Preprocessed_Selected_Features)
        CustomUtils.export_dataset(A_Labeled_Preprocessed_Selected_Features_path, A_Labeled_Dataset_Preprocessed_Selected_Features )
        CustomUtils.dict_to_yaml({'Selected Features':selected_features}, './Config/SelectedFeatures.yaml')

        ### D
        A_Labeled_Dataset_Preprocessed_Selected_Features = CustomUtils.import_dataset(file_path=A_Labeled_Preprocessed_Selected_Features_path)

        for i in range(3):
            optimized_hyperparameters = D_HyperparametersOptimization.optimize_hyperparameters(method_to_be_optimized=CustomUtils.methods[i])
            CustomUtils.dict_to_yaml(optimized_hyperparameters, './Config/OptimizedHyperparameters_'+CustomUtils.methods[i]+'.yaml')

        ### E
        A_Labeled_Dataset_Preprocessed_Selected_Feature = CustomUtils.import_dataset(file_path=A_Labeled_Preprocessed_Selected_Features_path)
        
        for i in range(3):
            model = E_Models_Generation.train_model(CustomUtils.methods[i], './Config/OptimizedHyperparameters_'+CustomUtils.methods[i]+'.yaml', A_Labeled_Dataset_Preprocessed_Selected_Features)#E_Models_Generation.select_model(0)
            CustomUtils.save_model(model, filename = '../Models/' + CustomUtils.methods[i]+'.pkl')

        #E_Models_Generation.voting_predict()
            
        ### F
        for i in range(3):
            F_Models_Validation.evaluate_model(A_Labeled_Preprocessed_Selected_Features_path, '../Models/' + CustomUtils.methods[i]+'.pkl')
            input('Continue...')

        '''A_Dataset_Generation.main()
        B_Dataset_ML_Friendly_Convertion.main()
        C_Recursive_Feature_Elimination.main()
        D_HyperparametersOptimization.main()
        E_Models_Generation.main()
        F_Models_Validation.main()'''
        
    elif selected_workflow == 'W2':
        ### G
        B_Unlabeled_Dataset = G_Dataset_Oversampling.ovesample_dataset(file_path=B_Unlabeled_path)
        dataset_info=CustomUtils.get_dataset_info(B_Unlabeled_Dataset)
        CustomUtils.export_dataset(B_Unlabeled_Oversampled_path, B_Unlabeled_Dataset)

        ### B
        B_Unlabeled_Dataset = CustomUtils.import_dataset(file_path=B_Unlabeled_path)
        B_Unlabeled_Dataset_Preprocessed = B_Dataset_ML_Friendly_Convertion.process_dataset(B_Unlabeled_Dataset)
        dataset_info=CustomUtils.get_dataset_info(B_Unlabeled_Dataset_Preprocessed)
        CustomUtils.export_dataset(A_Labeled_Preprocessed_path, A_Labeled_Dataset_Preprocessed )
    elif selected_workflow == 'W3':
        pass
    elif selected_workflow == 'W4':
        pass


    '''A_Dataset_Generation.main()
    input('\nPress [ENTER] to continue...\n')
    B_Dataset_ML_Friendly_Convertion.main()
    input('\nPress [ENTER] to continue...\n')
    C_Recursive_Feature_Elimination.main()
    input('\nPress [ENTER] to continue...\n')
    D_HyperparametersOptimization.main()
    input('\nPress [ENTER] to continue...\n')
    E_Models_Generation.main()
    input('\nPress [ENTER] to continue...\n')
    F_Models_Validation.main()
    input('\nPress [ENTER] to continue...\n')
    G_Labels_Prediction_Ba.main()
    input('\nPress [ENTER] to exit...\n')'''

if __name__ == "__main__":
    main()
