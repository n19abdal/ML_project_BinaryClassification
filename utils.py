#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 20:30:13 2021

@author: Fatima-Zahra KERDAD
@author: Imane Zriaa
@author: Emmanuel LEGUET
@author: Nada ABDALLAOUI

"""
# Import
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from pandas.core.frame import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# from IPython.display import display, HTML # coulb be useful in display_train_results, in a notebook
import plotly.graph_objects as go  # used only by pca_plotly_3d


### 1- Pretreatment

def load_data(path_file):
    """

    :param path_file: The path to the csv file of the data
    :return:  Loading the data using Pandas, return dataframe

    @Author : ZRIAA Imane
    """
    missing_values = ["?", "\t?"]  # Values to consider as NaN values
    df = pd.read_csv(path_file, na_values=missing_values)
    return df


def detect_type(df):
    """
    :param df: The dataframe of the dataset
    :return: Return the categorical and the numerical columns

    @Author : ZRIAA Imane
    """

    if 'id' in df.columns:
        df = df.drop(columns=['id'])  # Drop id column as it's not relevent for predictions

    cat_columns = list(df.select_dtypes(include=['object']).columns)  # get categorical columns

    for x in cat_columns:
        df[x] = df[x].str.replace('\t', '')
        df[x] = df[x].replace('?', np.nan)
        df[x] = df[x].str.replace(' ', '')
        df[x] = df[x].str.strip()  # Remove Blanks from the labels Column

    label_column = df.columns[-1]

    if label_column in cat_columns:
        cat_columns.remove(label_column)

    num_columns = [x for x in df.columns if
                   x not in cat_columns and x != label_column]  # get the numerical columns except the label

    return df, num_columns, cat_columns, label_column


def fill_na(df, cat_columns, num_columns):
    """

    :param df: dataframe with missing values
    :param cat_columns: categorical columns
    :param num_columns:  numerical columns
    :return: dataframe with zero missing values, the function replaces missing values in numerical values with the mean

    @Author : ZRIAA Imane
    """

    for x in num_columns:
        median = df[x].median()  # median of x column
        df[x] = df[x].fillna(median)  # replace the missing values by median

    for x in cat_columns:
        mode = df[x].mode()  # mode of x column
        df[x] = df[x].fillna(mode[0])  # replace the missing values by most frequent category

    return df


def transform_label(df, label_col):
    """

    :param df: DataFrame with no missing values
    :param label_col: The target of the data ( label)
    :return: the encoding of the label. We also drop the label from the dataframe

    @Author : ZRIAA Imane
    """

    le = LabelEncoder()
    le.fit(df[label_col])  # fit the labelEncoder
    label = le.transform(df[label_col])  # Encode the labels column
    df = df.drop([label_col], axis=1)  # Drop the categorical label column
    return df, label


def transform_data(df, cat_columns, num_columns):
    """

    :param df: the dataframe with non missing values
    :param cat_columns: categorical features to encode
    :param num_columns: numerical features to normalize
    :return: The transformed dataframe (after encoding and normalization)

    @Author : ZRIAA Imane
    """
    encoder = OrdinalEncoder()  # Encoder for categorical features
    scaler = StandardScaler()  # Encoder for standardization

    if len(cat_columns) != 0 and len(num_columns) != 0: #Dataframe with both categorical and numerical features
        categ_data = df[cat_columns]
        categ_transformed_data = encoder.fit_transform(categ_data)
        num_data = df[num_columns]
        num_transformed_data = scaler.fit_transform(num_data)
        df_transformed = pd.DataFrame(np.concatenate((num_transformed_data, categ_transformed_data), axis=1),
                                      columns=list(num_columns) + list(cat_columns))

    elif len(cat_columns) != 0: #DataFrame with only categorical features
        categ_data = df[cat_columns]
        categ_transformed_data = encoder.fit_transform(categ_data)
        df_transformed = pd.DataFrame(categ_transformed_data, columns=cat_columns)

    elif len(num_columns) != 0:  #DataFrame with only numerical features
        num_data = df[num_columns]
        num_transformed_data = scaler.fit_transform(num_data)
        df_transformed = pd.DataFrame(num_transformed_data, columns=num_columns)

    return df_transformed


### 2. Visualisation
def box_plot(dataframe):
    """

    :param dataframe: data frame
    :return: Box plot

    @Author : Fatima-Zahra KERDAD
    """
    num_col = detect_type(dataframe)[1]  # get numerical columns
    fig, axes = plt.subplots(nrows=1, ncols=len(num_col), figsize=(20, 10))
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        ax.boxplot(list(dataframe[num_col[i]]))
        ax.set_title(num_col[i])


def scatter_plot(dataframe):
    """

    :param dataframe: data frame
    :return: scatter plot

    @Author : Fatima-Zahra KERDAD
    """
    dataframe, num_col = detect_type(dataframe)[0], detect_type(dataframe)[1]
    list_colors = ['red', 'blue']  # Define colors for each class
    labels = list(set(dataframe[detect_type(dataframe)[3]]))  # Define labels

    fig, axes = plt.subplots(nrows=int(len(num_col) / 2), ncols=len(num_col) - 1, figsize=(30, 15))
    fig.tight_layout()

    plots = [(i, j) for i in range(len(num_col)) for j in range(len(num_col)) if i < j]

    for i, ax in enumerate(axes.flat):
        for j in range(2):
            x = dataframe.columns[plots[i][0]]
            y = dataframe.columns[plots[i][1]]
            ax.scatter(dataframe[dataframe[dataframe.columns[-1]] == labels[j]][x],
                       dataframe[dataframe[dataframe.columns[-1]] == labels[j]][y], color=list_colors[j])
            ax.set(xlabel=x, ylabel=y)

    fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0, 0.85))
    plt.show()


def bar_plot(dataframe):
    """

    :param dataframe: data frame
    :return: bar plot

    @Author : Fatima-Zahra KERDAD
    """
    cat_columns = detect_type(dataframe)[2]
    plt.subplots(figsize=(30, 50))

    for i in range(len(cat_columns)):
        plt.subplot(len(cat_columns), len(cat_columns), i + 1)
        dataframe[cat_columns[i]].value_counts().plot(kind='bar')
        plt.title(cat_columns[i])
    plt.show()


### 3- Feature selection


# 3.1 Correlation
def matrix_correlation_plot(dataframe):
    """
    :param dataframe: data frame
    :return: correlation matrix and plot it

    @Author : Fatima-Zahra KERDAD
    """

    corr_matrix = dataframe.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)

    # plot correlation matrix 
    plt.figure(figsize=(20, 20))
    corr_map = sns.diverging_palette(90, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=corr_map, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .6}, annot=True)
    plt.title('Correlation matrix')
    plt.show()

    return corr_matrix


def data_correlation(dataframe, corr_threshold):
    """
    :description: Features with high correlation are more linearly dependent
    so they have almost the same effect. That's why it important to drop one of
    two features that have high correlation

    :param dataframe: data frame
    :param corr_threshold: threshold of correlation
    :return: Compare the correlation between features
    and drop one of two features that have a correlation higher than threshold
    
    @Author : Fatima-Zahra KERDAD
    """

    # Compute and plot matrix correlation
    # corr_matrix = dataframe.corr()
    if 'id' in dataframe.columns:
        dataframe = dataframe.drop(columns=['id'])

    corr_matrix = dataframe.corr()

    # drop one of two feature that have correlation higher than the threshold
    num_columns = np.array(detect_type(dataframe)[1])
    columns_bool = np.full((len(num_columns),), True, dtype=bool)

    for i in range(len(columns_bool)):
        for j in range(len(columns_bool)):
            if i != j:
                if abs(corr_matrix.iloc[i, j]) >= corr_threshold:
                    if columns_bool[j]:
                        columns_bool[j] = False

    return num_columns[columns_bool]


# 3.2 PCA

def apply_pca(df_transformed: DataFrame, n_components: int = None, explained_variance: float = None) -> np.array:
    """

    :param df_transformed: the transformed dataframe, centered and normalized 
    :param n_components: number of principal components to keep (priority on the explained variance)
    :param explained_variance: percentage of explained variance wanted to select the principal components
    :return: the array with the input data projected on the principal components kept

    @Author: Emmanuel LEGUET
    """
    pca = PCA(n_components or explained_variance)
    pca.fit(df_transformed)
    reduced_data = pca.transform(df_transformed)
    return reduced_data


# PCA Visualisation
def pca_vizualize_2d(df_clean: DataFrame, df_transformed: DataFrame, col: str) -> None:
    """

    :param df_clean: the cleaned dataframe
    :param df_transformed: the transformed dataframe, centered and normalized
    :param col: the column to use to label the data

    @Author: Emmanuel LEGUET
    """
    # PCA with 2 components
    X = apply_pca(df_transformed, 2)

    # Plot
    plt.figure(figsize=(8, 8))

    plt.title(f"'{col}' through 2 components PCA", fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)

    plt.scatter(X[:, 0], X[:, 1], c=df_clean[col])

    plt.legend()
    plt.grid()
    plt.show()


def pca_plotly_3d(df_clean: DataFrame, df_transformed: DataFrame, col: str) -> None:  # ! Using plotly
    """

    :param df_clean: the cleaned dataframe
    :param df_transformed: the transformed dataframe, centered and normalized
    :param col: the column to use to label the data

    @Author: Emmanuel LEGUET
    """
    # PCA with 3 components
    X = apply_pca(df_transformed, 3)

    # Define layout
    layout = go.Layout(
        autosize=True,
        showlegend=True,
        title=f"Visualization of '{col}' with 3 components PCA ",
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='PC 1'),
            yaxis=go.layout.scene.YAxis(title='PC 2'),
            zaxis=go.layout.scene.ZAxis(title='PC 3')
        )
    )

    # Define figure
    fig = go.Figure(data=[go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(
        size=3,
        color=df_clean[col],
        opacity=0.9,
    ))],
                    layout=layout,
                    )

    # Plot figure
    fig.show()


def pca_plot_3d(df_clean: DataFrame, df_transformed: DataFrame, col: str) -> None:  # Using matplotlib
    """

    :param df_clean: the cleaned dataframe
    :param df_transformed: the transformed dataframe, centered and normalized
    :param col: the column to use to label the data

    @Author: Emmanuel LEGUET
    """
    # PCA with 3 components
    X = apply_pca(df_transformed, 3)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Visualization of '{col}' with 3 components PCA ")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=df_clean[col], marker='o')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    plt.show()


### 4- Training Phase

def split_data(df_transformed, labels, test_size):
    """

    :param df: the original dataframe cleaned
    :param df_transformed: the transformed dataframe, centered and normalized, without the label column
    :param test_size: size of our split
    :return: the training and the test set

    @Author: Nada ABDALLAOUI
    """

    X_train, X_test, y_train, y_test = train_test_split(df_transformed, labels, test_size=test_size)
    return X_train, X_test, y_train, y_test


def train_models(models: dict, X_train: np.array, y_train: np.array, **gs_kwargs) -> dict:
    """Apply gridsearch with cross-validation to optimize the input models and train the best estimator for each model.

    :param models: models to train, their name and parameters to try
        -> Example:
            dict(
                knn = dict(
                    'model':KNeighborsClassifier, 
                    'parameters':{'n_neighbors':[5,6,8,10,12,14,15]}),
                ),
                tree = dict(
                    'model':DecisionTreeClassifier, 
                    'parameters':{'criterion':['gini','entropy'], 'max_depth':[5,7,9,10]}
                ),
            )
    :param X_train: training data
    :param y_train: training label
    :return: results from the grid searches, containing the best trained estimators

    @Author: Emmanuel LEGUET
    """
    print("Parameters optimization & Estimator training:")
    gscv_res = dict()
    pbar = tqdm(models.items())
    for name, m in pbar:
        pbar.set_description(f"Processing {name} model")
        gscv = GridSearchCV(m['model'](), m['parameters'], refit=True, **gs_kwargs)  # n_jobs=-1 to use all processors
        gscv_res[name] = gscv.fit(X_train, y_train)

    return gscv_res


def display_train_results(gscv_res: dict, full: bool = False) -> None:
    """Display grid search results and the scores of the best estimators.

    :param gscv_res: grid searches result obtained with train_pipeline
    :param full: if False, only display the  scores of the best estimators

    @Author: Emmanuel LEGUET
    """
    # Display grid search results
    if full:
        for name, gscv in gscv_res.items():
            print(f"\n{name} model:")
            print(pd.DataFrame(gscv.cv_results_).to_markdown())
            # display(HTML(pd.DataFrame(gscv.cv_results_).to_html())) # -> with IPython

    # Display score & parameters for best estimator of each model
    print("\n\n Score results for best etimators:")
    df_scores = pd.DataFrame({'Model': gscv_res.keys(),
                              'Score': [g.best_score_ for g in gscv_res.values()],
                              'Parameters': [g.best_params_ for g in gscv_res.values()]
                              })
    print(df_scores.sort_values(by="Score", ascending=False).to_markdown())
    # display(HTML(df_scores.sort_values(by="Score", ascending=False).to_html())) # -> with IPython


def evaluate_model(models, test_data, label):
    """
    :description: Evaluate the model on test data
        
    :param model: models to evaluate 
    :param data: test set to predict 
    :param label: true labels
    :return:  f1_score, accuracy, precision, recall for each model

    @Author : Fatima-Zahra KERDAD
    """
    evaluation = {}
    for model in models.keys():
        eval = {}
        y_pred = models[model].predict(test_data)
        precision = precision_score(label, y_pred)
        accuracy = accuracy_score(label, y_pred)
        recall = recall_score(label, y_pred)
        f1_Score = f1_score(label, y_pred, average='macro')
        # clf_report = classification_report(label, y_pred)
        eval["f1 score"] = f1_Score
        eval["accuracy"] = accuracy
        eval["precision"] = precision
        eval["recall"] = recall
        # eval["report"] = clf_report
        evaluation[model] = eval

    return evaluation
