# Predict Customer Churn
- Author: Bernardo Carvalho
- Date: July 2022

<l>
A manager at the bank is disturbed by more and more customers leaving their credit card services. They would appreciate it if one could predict who is going to get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.
<l> 

## Project Description
This package tries to provide a solution to predict what client is more likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). 

The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

<l>

## Files and data description
Overview of the files and data present in the root directory. 

```bash
├── churn_notebook.ipynb # Original exploratory notebook that was refactored
├── churn_library.py     # Main module with the defined functions
├── churn_script_logging_and_tests.py # Tests and logs of the main module (churn_library.py)
├── requirements.txt     # Python requirements for the project
├── constants.py         # Module with stored constants
├── README.md            # Project overview, and instructions to use the code
├── data                 # Folder with the data used in this project
│   └── BankChurners.csv
├── images               # EDA plots and classification results
│   ├── eda
│   └── results
├── logs                 # logs
└── models               # models
```
<l>

## Dependencies
The only dependency is python in the following version:
```
Python == 3.8
```

## Requirements

Run in your terminal the following code line so you can install all the required libraries.
```
- pip install -r requirements_py.txt
```
<l>

## Running Files
How do you run your files? What should happen when you run your files?

- churn_library.py

The churn_library.py is the main module that will recreate the EDA and the feature engineer following the modeling step.

```
ipython churn_library.py
```
This command will produce:
- EDA files in ./images/eda/;
- Result analysis of each model ./images/results/;
- Two stored models in ./models/;

- churn_script_logging_and_tests.py

If you want to test the churn_library, run the following command via terminal.
```
ipython churn_script_logging_and_tests.py
```
This command will produce a test and logging of the main modulo churn_library.py.



