"""
Module to predict customer churn

Author: Bernardo C.
Date: 2022/06/20

"""

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as ct
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    frame = pd.read_csv(pth)
    return frame


def perform_eda(frame: pd.DataFrame):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # How many null values are present on each column
    null_df = frame.isnull().sum().to_frame().reset_index().rename(
        columns={'index': 'Columns Names', 0: 'Null_total'})
    null_df.to_csv(ct.EDA_PATH + 'null_values.csv', sep=',')

    # The description of each column in the dataframe
    summary_df = frame.describe()
    summary_df.to_csv(ct.EDA_PATH + 'desc_data.csv', sep=',')

    # Eda Figures

    frame['Churn'] = frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    frame['Churn'].hist()
    plt.savefig(ct.EDA_PATH + 'churn_distribution.png')

    plt.figure(figsize=(20, 10))
    frame['Customer_Age'].hist()
    plt.savefig(ct.EDA_PATH + 'customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    frame['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(ct.EDA_PATH + 'martial_status_distribution.png')

    plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth
    #  curve obtained using a kernel density estimate
    sns.histplot(frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(ct.EDA_PATH + 'total_transaction_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(ct.EDA_PATH + 'heatmap.png')


def encoder_helper(frame: pd.DataFrame,
                   category_lst: list,
                   response: str = None
                   ):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used
         for naming variables or index y column]
    output:
        df: pandas dataframe with new columns for
    '''

    # Creationg the churn target variable
    frame['Churn'] = frame['Attrition_Flag'].apply(
        lambda value: 0 if value == "Existing Customer" else 1)

    def categorical_conversion(frame: pd.DataFrame, feature: str):
        """
        Function to generate new features from categorical ones using the proportion
        of churn for each category

        input:
            df: (pandas dataframe)
            feature: (str) categorical feature name
        output:
            df: (pandas dataframe) with the new created feature

        """

        # Proportion of the feature by the mean churn
        grouped_value = frame.groupby(feature).mean()['Churn']
        feature_value_list = []
        # Interation over reach category value and append
        #  the respective value to a list
        for value in frame[feature]:
            feature_value_list.append(grouped_value.loc[value])
        return feature_value_list

    # Running the categorical_conversion over all the categorical inputed
    # features
    for categorical_feature in category_lst:
        if response is None:
            frame[categorical_feature +
                  '_Churn'] = categorical_conversion(frame, categorical_feature)
        else:
            frame[categorical_feature +
                  response] = categorical_conversion(frame, categorical_feature)

    control_frame = frame.copy()
    # Return the transformed features and removing the categorical ones
    control_frame.drop(category_lst, axis=1, inplace=True)
    return control_frame


def perform_feature_engineering(frame: pd.DataFrame):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument
                that could be used for naming variables or index y column]

    output:
              features_train: X training data
              features_test: X testing data
              target_train: y training data
              features_test: y testing data
    '''
    # Categorical columns to use on encoder helper
    category_cols = ['Gender',
                     'Education_Level',
                     'Marital_Status',
                     'Income_Category',
                     'Card_Category']

    encoded_frame = encoder_helper(frame, category_lst=category_cols)
    features_data = encoded_frame[ct.KEEP_COLS].copy()
    target_data = encoded_frame['Churn']
    features_train, features_test, target_train, target_test = train_test_split(
        features_data, target_data, test_size=0.3, random_state=42)
    return features_train, features_test, target_train, target_test


def classification_report_image(target_train,
                                target_test,
                                target_train_pred_lr,
                                target_train_pred_rf,
                                target_test_pred_lr,
                                target_test_pred_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            targetTrain: training response values
            targetTest:  test response values
            targetTrain_pred_lr: training predictions from logistic regression
            targetTrain_pred_rf: training predictions from random forest
            targetTest_pred_lr: test predictions from logistic regression
            targetTest_pred_rf: test predictions from random forest

    output:
             None
    '''
    def classification_function(
            true_target: tuple,
            predict_target: tuple,
            model_name: str):

        target_train, target_test = true_target
        predict_target_train, predict_target_test = predict_target

        plt.rc('figure', figsize=(7, 7))
        plt.text(0.01, 1.25, str(f'{model_name} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    target_test, predict_target_test)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model_name} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    target_train, predict_target_train)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        model_name_to_save = model_name.replace(' ', '_').lower()
        plt.savefig(
            ct.RESULTS_PATH +
            f'{model_name_to_save}_classification_report.png')

    # Random Forest Reports
    classification_function(
        true_target=(target_train, target_test),
        predict_target=(target_train_pred_rf, target_test_pred_rf),
        model_name='Random Forest')

   # Logistic Regression Reports
    classification_function(
        true_target=(target_train, target_test),
        predict_target=(target_train_pred_lr, target_test_pred_lr),
        model_name='Logistic Regression')


def feature_importance_plot(model, features_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(features_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(features_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(features_train,
                 features_test,
                 target_train,
                 target_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    def random_forest(features_train,
                      features_test,
                      target_train):

        # Random Forest Classifier
        rfc = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        # Grid Search parameters
        cv_rfc = GridSearchCV(estimator=rfc,
                              param_grid=param_grid,
                              cv=5)

        cv_rfc.fit(features_train, target_train)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(features_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(features_test)
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

        return y_train_preds_rf, y_test_preds_rf, cv_rfc

    def logistic_regression(features_train,
                            features_test,
                            target_train):

        # Logistic Regression model
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        lrc.fit(features_train, target_train)

        y_train_preds_lr = lrc.predict(features_train)
        y_test_preds_lr = lrc.predict(features_test)

        joblib.dump(lrc, './models/logistic_model.pkl')

        return y_train_preds_lr, y_test_preds_lr, lrc

    target_train_pred_rf, target_test_pred_rf, model_rf = random_forest(
        features_train,
        features_test,
        target_train
    )

    target_train_pred_lr, target_test_pred_lr, model_lr = logistic_regression(
        features_train,
        features_test,
        target_train
    )

    classification_report_image(target_train=target_train,
                                target_test=target_test,
                                target_train_pred_lr=target_train_pred_lr,
                                target_train_pred_rf=target_train_pred_rf,
                                target_test_pred_lr=target_test_pred_lr,
                                target_test_pred_rf=target_test_pred_rf)

    feature_importance_plot(model=model_rf,
                            features_data=features_train,
                            output_pth=ct.RESULTS_PATH +
                            "feature_importance_rf.png")

    def roc_curve_comparative(model_1, model_2, features_test, target_test):
        plt.figure(figsize=(15, 8))
        axis = plt.gca()
        plot_roc_curve(model_1, features_test, target_test, ax=axis, alpha=0.8)
        plot_roc_curve(model_2, features_test, target_test, ax=axis, alpha=0.8)
        plt.savefig(ct.RESULTS_PATH + 'roc_curve_result.png')

    roc_curve_comparative(
        model_lr,
        model_rf.best_estimator_,
        features_test,
        target_test)


if __name__ == '__main__':
    df = import_data(ct.DATA_PATH)
    perform_eda(df)
    features_tr, features_te, target_tr, target_te = perform_feature_engineering(
        df)
    train_models(features_tr, features_te, target_tr, target_te)
