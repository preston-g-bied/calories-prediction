"""
Data validation utilities for the Calories Prediction project.
This module provides functions to validate datasets and features.
"""

import pandas as pd
import numpy as np
import logging

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing value counts and percentages
    """
    logger.info("Checking for missing values")

    # calculate missing values count and percentage
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    # create dataframe with results
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    }).sort_values('Missing Count', ascending=False)

    # filter only columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if len(missing_df) == 0:
        logger.info("No missing values found")
    else:
        logger.warning(f"Found missing values in {len(missing_df)} columns")

    return missing_df

def check_data_types(train_df, test_df):
    """
    Check for data type consistency between train and test datasets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with data type comparison
    """
    logger.info("Checking data type consistency")

    # get common columns
    common_cols = list(set(train_df.columns) & set(test_df.columns))

    # create dataframe with data types
    dtype_df = pd.DataFrame({
        'Train': train_df[common_cols].dtypes,
        'Test': test_df[common_cols].dtypes
    })

    # check for inconsistencies
    dtype_df['Match'] = dtype_df['Train'] == dtype_df['Test']
    inconsistent = dtype_df[~dtype_df['Match']]

    if len(inconsistent) == 0:
        logger.info("All data types are consistent between train and test datasets")
    else:
        logger.warning(f"Found {len(inconsistent)} columns with inconsistent data types")

    return dtype_df

def check_value_ranges(train_df, test_df):
    """
    Check for value range consistency between train and test datasets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with value range comparison
    """
    logger.info("Checking value range consistency")

    # get common numerical columns
    common_num_cols = list(
        set(train_df.select_dtypes(include=['number']).columns) &
        set(test_df.select_dtypes(include=['number']).columns)
    )

    # exclude id column if present
    if 'id' in common_num_cols:
        common_num_cols.remove('id')

    # create results dataframe
    results = []

    for col in common_num_cols:
        train_min = train_df[col].min()
        train_max = train_df[col].max()
        test_min = test_df[col].min()
        test_max = test_df[col].max()

        # check if test values are outside training range
        min_outside = test_min < train_min
        max_outside = test_max > train_max

        results.append({
            'Feature': col,
            'Train_Min': train_min,
            'Train_Max': train_max,
            'Test_Min': test_min,
            'Test_Max': test_max,
            'Min_Outside_Range': min_outside,
            'Max_Outside_Range': max_outside,
            'Outside_Range': min_outside or max_outside
        })

    results_df = pd.DataFrame(results)

    outside_range = results_df[results_df['Outside_Range']]
    if len(outside_range) == 0:
        logger.info("All test values are within the training data range")
    else:
        logger.warning(f"Found {len(outside_range)} features with test value outside training range")

    return results_df

def check_feature_correlation(df, target='Calories', threshold=0.01):
    """
    Check feature correlation with target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target : str
        Target variable name
    threshold : float
        Correlation threshold to highlight low correlation
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature correlations
    """
    logger.info(f"Checking feature correlation with {target}")

    # ensure target is in the dataframe
    if target not in df.columns:
        logger.error(f"Target '{target}' not found in dataframe")
        return None
    
    # get numerical features
    num_features = df.select_dtypes(include=['number']).columns.tolist()
    if target in num_features:
        num_features.remove(target)
    if 'id' in num_features:
        num_features.remove('id')

    # calculate correlation with target
    correlations = df[num_features + [target]].corr()[target].drop(target)
    abs_correlations = correlations.abs().sort_values(ascending=False)

    # create dataframe with results
    corr_df = pd.DataFrame({
        'Feature': abs_correlations.index,
        'Correlation': correlations[abs_correlations.index],
        'Abs_Correlation': abs_correlations.values
    })

    # highlight low correlation features
    low_corr = corr_df[corr_df['Abs_Correlation'] < threshold]
    if len(low_corr) > 0:
        logger.warning(f"Found {len(low_corr)} features with correlation < {threshold}")

    return corr_df

def validate_datasets(train_df, test_df, target='Calories'):
    """
    Run all validation checks on datasets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    target : str
        Target variable name
    
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    logger.info("Running all validation checks")

    results = {}

    # check missing values
    results['missing_train'] = check_missing_values(train_df)
    results['missing_test'] = check_missing_values(test_df)

    # check data types
    results['data_types'] = check_data_types(train_df, test_df)

    # check value ranges
    results['value_ranges'] = check_value_ranges(train_df, test_df)

    # check feature correlation
    if target in train_df.columns:
        results['correlations'] = check_feature_correlation(train_df, target=target)

    return results