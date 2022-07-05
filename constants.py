# Data Path
DATA_PATH = "./data/BankChurners.csv"

# Exploratory data analysis saving path
EDA_PATH = "./images/eda/"

EDA_FILES = ['null_values.csv',
             'desc_data.csv',
             'churn_distribution.png',
             'customer_age_distribution.png',
             'total_transaction_distribution.png',
             'martial_status_distribution.png',
             'heatmap.png']


# Resulting figure files from the train test function
RESULT_FIG_PATH = './images/results/'
RESULT_FIG_FILES = ['feature_importance_rf.png',
                    'logistic_regression_classification_report.png',
                    'random_forest_classification_report.png',
                    'roc_curve_result.png']

# Model saved object
RESULT_MODEL_PATH = './models/'
RESULT_MODEL_FILES = ['logistic_model.pkl',
                      'rfc_model.pkl']

# Categorical columns to use on encoder helper
CATEGORY_COLS = ['Gender',
                 'Education_Level',
                 'Marital_Status',
                 'Income_Category',
                 'Card_Category']

# Features to train
KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit',
             'Total_Revolving_Bal', 'Avg_Open_To_Buy',
             'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
             'Avg_Utilization_Ratio', 'Gender_Churn',
             'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

# Result plots path
RESULTS_PATH = "./images/results/"
