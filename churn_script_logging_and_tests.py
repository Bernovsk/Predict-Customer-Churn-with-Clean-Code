"""
Module to test the churn_library.py

Author: Bernardo C.
Date: 2022/07/04

"""

import os
import logging
import pandas as pd
import churn_library as clib
import constants as ct

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def cleaning_files(dir_path, files_list):
    """
   Checks if the files are in the selected directory and remove them.
    """
    logging.info('REMOVING OLD FILES INSIDE')
    for file in files_list:
        if os.path.exists(dir_path + file):
            logging.info('File %s was found: SUCCESS', file)
            os.remove(dir_path + file)
        else:
            logging.info('File %s not founded: FAIL', file)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        frame = pd.read_csv(ct.DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")


def test_eda():
    '''
    Test perform eda function
    '''

    cleaning_files(ct.EDA_PATH, ct.EDA_FILES)

    try:
        frame = clib.import_data(ct.DATA_PATH)
        clib.perform_eda(frame)
        logging.info('Perform EDA: SUCCESS')
    except Exception:
        logging.error('Perform EDA: FAIL')

    create_files = os.listdir(ct.EDA_PATH)
    try:
        assert sorted(ct.EDA_FILES) == sorted(create_files)
        logging.info('Testing perform eda: SUCCESS')
        for file in create_files:
            logging.info(
                "The file %s was created",
                file.replace(
                    '.csv', ''))

    except FileNotFoundError:
        missing_files = [*set(ct.EDA_FILES) - set(create_files)]
        for miss_file in missing_files:
            logging.error(
                'File %s could not be found',
                miss_file.replace(
                    '.csv', ''))


def test_encoder_helper():
    '''
    Test encoder helper
    '''
    frame = clib.import_data(ct.DATA_PATH)

    frame_conversion = clib.encoder_helper(
        frame, ct.CATEGORY_COLS, response=None)

    try:
        assert 'Churn' in frame_conversion.columns
        logging.info('The Churn column was created: SUCCESS')
    except AssertionError:
        logging.error('Churn column was not created: FAIL')

    transform_features_names = [
        cat_col + '_Churn' for cat_col in ct.CATEGORY_COLS]
    created_columns = [new_features for new_features in frame_conversion.columns.tolist(
    ) if '_Churn' in new_features]
    try:
        assert sorted(transform_features_names) == sorted(created_columns)
        logging.info('Encoder Helper: SUCCESS')

    except AssertionError:
        logging.error('Encoder Helper: FAIL')
        for expected, created in zip(sorted(transform_features_names),
                                     sorted(created_columns)):

            logging.info('%s - %s', expected, created)


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    frame = clib.import_data(ct.DATA_PATH)
    frame_rows_size = frame.shape[0]
    train_real_size = int(frame_rows_size * 0.7)
    test_real_size = frame_rows_size - train_real_size

    try:
        logging.info('TEST: Running perform feature engineering')
        features_train, features_test, target_train, target_test = clib.perform_feature_engineering(
            frame)
        logging.info('TEST: Perform feature engineering: SUCCESS')
    except Exception:
        logging.error('TEST: Perform feature engineering: FAIL')

    try:
        logging.info('TEST: Perform feature engineering train output')
        assert (
            features_train.shape[0] == train_real_size) & (
            target_train.shape[0] == train_real_size)
        logging.info('Train size: SUCCESSS')
    except AssertionError:
        logging.error('Train size is wrong: FAIL')

    try:
        logging.info('TEST: Perform feature engineering train output')
        assert (
            features_test.shape[0] == test_real_size) & (
            target_test.shape[0] == test_real_size)
        logging.info('Test size: SUCCESSS')
    except AssertionError:
        logging.error('Test size is wrong: FAIL')


def test_train_models():
    '''
    test train_models
    '''
    cleaning_files(ct.RESULT_FIG_PATH, ct.RESULT_FIG_FILES)
    cleaning_files(ct.RESULT_MODEL_PATH, ct.RESULT_MODEL_FILES)

    try:
        logging.info('Running necessary functions for train_models')
        logging.info('Importing the Data')
        frame = clib.import_data(ct.DATA_PATH)
        logging.info('Import data: SUCCESS')
        logging.info('Perfoming the feature engineering')
        features_train, features_test, target_train, target_test = clib.perform_feature_engineering(
            frame)
        logging.info('Perform feature engineering: SUCCESS')

    except Exception:
        logging.error('Run necessary functions: FAIL')

    try:
        logging.info('Run train_models')
        clib.train_models(
            features_train,
            features_test,
            target_train,
            target_test)
        logging.info('Test train_models: SUCCESS')
    except Exception:
        logging.error('Test train_models: FAIL')

    created_fig_files = os.listdir(ct.RESULT_FIG_PATH)
    created_models_files = os.listdir(ct.RESULT_MODEL_PATH)

    def verify_files(expected_files, dir_files):
        """
        Checks if the expected file was created.
        """
        try:
            assert os.path.exists(dir_files + expected_files)
            logging.info('Testing perform eda: SUCCESS')
            logging.info(
                "The file %s was created",
                expected_files)
        except FileNotFoundError:
            logging.error(
                'File %s could not be found', expected_files)

    for fig_file in created_fig_files:
        verify_files(fig_file, ct.RESULT_FIG_PATH)

    for model_file in created_models_files:
        verify_files(model_file, ct.RESULT_MODEL_PATH)


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
