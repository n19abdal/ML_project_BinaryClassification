from utils import *
import os
from pathlib import Path
import argparse 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
# models
models = dict(
            gauss={'model':GaussianNB, 'parameters':{}},
            kernel={'model':SVC,'parameters':{'kernel':['linear', 'rbf', 'sigmoid', 'poly'],'C':[1, 10], 
                                            'degree': [2, 3],'gamma' : ['scale', 'auto'] }},
            logisticregression={'model':LogisticRegression,'parameters':{'C':[1, 10],'fit_intercept' : [True,False],'intercept_scaling' : [1,10]}},
            knn={'model':KNeighborsClassifier, 'parameters':{'n_neighbors':[5,6,8,10,12,14,15]}}, 
            tree={'model':DecisionTreeClassifier, 'parameters':{'criterion':['gini','entropy'],'max_depth':[5,7,9,10],'min_samples_leaf':[1,2]}},
            adaboost={'model':AdaBoostClassifier, 'parameters':{'learning_rate':[0.1,0.001],'n_estimators':[100,250]}},
            gradientboost={'model':GradientBoostingClassifier, 'parameters':{'max_depth':[5,10],'min_samples_leaf':[1],'n_estimators':[100]}},
            sgdclassifier={'model':SGDClassifier,'parameters':{'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],'penalty':['l1', 'l2'], 'fit_intercept' : [True,False]}},
                                            
)

path_files = ["../Data/kidney_disease.csv","../Data/BankNoteAuthentication.csv"]

def cmd():
    """
    :description: manipulate argument parser

    :return: parser

    @Author : Fatima-Zahra KERDAD
    """
    
    parser = argparse.ArgumentParser(description='Main-Code') 
    parser.add_argument('--file', '-f', type=int, default=1,help = "Type 1 for kidney_disease file or 2 for BankNoteAuthentication")
    
    return parser

def load_preprocess(path_file):
    """
    :description: Loading and preprocessing the data
        
    :param path_file: The path to the csv file of the data
    :return: data frame after preprocessing

    @Author : Fatima-Zahra KERDAD
    """
    data_frame = load_data(path_file)
    data_frame, num_col, cat_col, label_col = detect_type(data_frame)
    data_frame = fill_na(data_frame, cat_col, num_col)
    data_frame, label = transform_label(data_frame, label_col)
    data_frame = transform_data(data_frame, cat_col, num_col)
    return data_frame, label


def main():
    """
    :description: main function
    
    @Author : Fatima-Zahra KERDAD
    """
    parser = cmd() 
    args = parser.parse_args()
    path  = args.file
    path_file = path_files[path-1]
    data_frame, labels= load_preprocess(path_file)
    reduced_data_frame = apply_pca(data_frame, explained_variance=0.9)
    X_train, X_test, y_train, y_test =split_data(reduced_data_frame, labels, 0.7)
    gscv1 = train_models(models, X_train, y_train, n_jobs=-1)
    display_train_results(gscv1)
    evaluations = evaluate_model(gscv1, X_test, y_test)
    for mod in evaluations.keys():
        print(f'{mod} :\n {evaluations[mod]}')

if __name__ == "__main__":
    main()