"""
Feature engineering functions for calories prediction model.
This module provides functions to create, transform, and select features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging

from src.data.validation import validate_datasets

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def align_datasets(train_df, original_df):
    """
    Align column names between training and original datasets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    original_df : pandas.DataFrame
        Original dataset
    
    Returns:
    --------
    tuple
        Aligned training and original datasets
    """
    logger.info("Aligning datasets")

    # create copies to avoid modifying the original dataframes
    train_copy = train_df.copy()
    original_copy = original_df.copy()

    # rename columns in original dataset to match training dataset
    original_copy = original_copy.rename(columns={
        'User_ID': 'id',
        'Gender': 'Sex'
    })

    # select only common columns
    common_columns = [col for col in train_copy.columns if col in original_copy.columns]

    logger.info(f"Common columns: {common_columns}")

    return train_copy, original_copy[common_columns]

def combine_datasets(train_df, original_df):
    """
    Combine training and original datasets.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    original_df : pandas.DataFrame
        Original dataset
    
    Returns:
    --------
    pandas.DataFrame
        Combined dataset
    """
    logger.info("Combining datasets")

    # align datasets first
    train_aligned, original_aligned = align_datasets(train_df, original_df)

    # combine datasets
    combined_df = pd.concat([train_aligned, original_aligned], axis=0, ignore_index=True)

    logger.info(f"Combined dataset shape: {combined_df.shape}")

    return combined_df

def create_basic_features(df):
    """
    Create basic features from raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with new features
    """
    logger.info("Creating basic features")

    # create a copy to avoid modifying the original dataframe
    df_features = df.copy()

    # BMI feature
    df_features['BMI'] = df_features['Weight'] / ((df_features['Height'] / 100) ** 2)

    # estimated resting heart rate based on age (220 - age)
    df_features['Rest_Heart_Rate'] = 220 - df_features['Age']

    # workout intensity metrics
    df_features['Heart_Rate_Reserve'] = df_features['Heart_Rate'] / df_features['Rest_Heart_Rate']
    df_features['Workout_Volume'] = df_features['Duration'] * df_features['Heart_Rate']

    # age categories
    df_features['Age_Category'] = pd.cut(
        df_features['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=['Young', 'Middle_Age', 'Senior', 'Elderly']
    )

    # body temp interactions
    df_features['Temp_Duration'] = df_features['Body_Temp'] * df_features['Duration']
    df_features['Temp_Heart_Rate'] = df_features['Body_Temp'] * df_features['Heart_Rate']

    # weight related features
    df_features['Weight_Duration'] = df_features['Weight'] * df_features['Duration']

    # sex as numeric (for some models)
    df_features['Sex_Numeric'] = df_features['Sex'].map({'male': 1, 'female': 0})

    logger.info(f"Created features: {list(set(df_features.columns) - set(df.columns))}")

    return df_features

def create_polynomial_features(df, degree=2, features=None):
    """
    Create polynomial features for specified columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    degree : int
        Degree of polynomial features
    features : list
        List of features to create polynomial features for. If None, uses default set.
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with polynomial features
    """
    logger.info(f"Creating polynomial features with degree={degree}")

    # create a copy to avoid modifying the original dataframe
    df_poly = df.copy()

    # default features if none specified
    if features is None:
        features = ['Duration', 'Heart_Rate', 'Body_Temp']

    # create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df_poly[features])

    # get feature names
    feature_names = poly.get_feature_names_out(features)

    # create a dataframe with polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names)

    # drop the original features from poly_df to avoid duplication
    for feature in features:
        if feature in poly_df.columns:
            poly_df = poly_df.drop(columns=[feature])

    # reset index to ensure proper concatenation
    df_poly = df_poly.reset_index(drop=True)
    poly_df = poly_df.reset_index(drop=True)

    # concatenate with original dataframe
    df_poly = pd.concat([df_poly, poly_df], axis=1)

    logger.info(f"Created {poly_df.shape[1]} polynomial features")

    return df_poly

def transform_features(df):
    """
    Apply transformations to features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with transformed features
    """
    logger.info("Applying feature transformations")

    # create a copy to avoid modifying original dataframe
    df_transformed = df.copy()

    # log-transform right-skewed numerical features
    # based on EDA, most features are normally distributed
    # Calories is right-skewed, which is relevant for RMSLE

    if 'Calories' in df_transformed.columns:
        df_transformed['Log_Calories'] = np.log1p(df_transformed['Calories'])

    # for features with high values
    df_transformed['Log_Workout_Volume'] = np.log1p(df_transformed['Workout_Volume'])

    logger.info(f"Applied transformations, new features: {list(set(df_transformed.columns) - set(df.columns))}")

    return df_transformed

def select_features(df, target='Calories', method='importance', k=15):
    """
    Select most important features using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset with features
    target : str
        Target variable name
    method : str
        Method to use for feature selection ('importance', 'kbest', 'rfe')
    k : int
        Number of features to select
    
    Returns:
    --------
    tuple
        Selected features dataframe, list of selected feature names
    """
    logger.info(f"Selecting features using {method} method")

    # create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # ensure target is in the dataframe
    if target not in df_copy.columns:
        raise ValueError(f"Target '{target}' not found in dataframe")

    # convert categorical features to numeric for feature selection
    X = df_copy.drop(columns=[target])

    # handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != 'Sex':    # special handling for Sex as it's already handled
            X = pd.get_dummies(X, columns=[col], drop_first=True)

    # make sure all data is numeric
    X = X.select_dtypes(include=['number'])
    y = df_copy[target]

    selected_features = []

    if method == 'kbest':
        # select top k features using f_regression (for regression tasks)
        selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        selector.fit(X, y)

        # get selected feature names
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)

        selected_features = feature_scores.head(k)['Feature'].tolist()

    elif method == 'rfe':
        # recursive feature elimination
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]), step=1)
        selector.fit(X, y)

        # get selected feature names
        selected_features = X.columns[selector.support_].tolist()

    elif method == 'importance':
        # use a tree-based model for feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=7)
        model.fit(X, y)

        # get feature importances
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        selected_features = feature_importances.head(k)['Feature'].tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Selected {len(selected_features)} features: {selected_features}")

    # return selected features
    X_selected = X[selected_features]

    return X_selected, selected_features

def check_multicollinearity(df, threshold=0.85):
    """
    Check for multicollinearity between features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    threshold : float
        Correlation threshold to identify high multicollinearity
    
    Returns:
    --------
    pandas.DataFrame
        Highly correlated feature pairs
    """
    logger.info(f"Checking multicollinearity with threshold={threshold}")

    # calculate correlation matrix
    corr_matrix = df.select_dtypes(include=['number']).corr().abs()

    # create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find feature pairs with correlation greater than threshold
    high_corr = [(upper.columns[i], upper.columns[j], upper.iloc[i, j])
                 for i in range(len(upper.columns))
                 for j in range(len(upper.columns))
                 if i < j and upper.iloc[i, j] > threshold]

    # convert to dataframe
    high_corr_df = pd.DataFrame(high_corr, columns=['Feature1', 'Feature2', 'Correlation'])
    high_corr_df = high_corr_df.sort_values('Correlation', ascending=False)

    logger.info(f"Found {len(high_corr_df)} highly correlated feature pairs")

    return high_corr_df

def remove_multicollinear_features(df, target=None, threshold=0.85):
    """
    Remove features with high multicollinearity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target : str
        Target variable name (will not be removed)
    threshold : float
        Correlation threshold to identify high multicollinearity
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with multicollinear features removed
    """
    logger.info(f"Removing multicollinear features with threshold={threshold}")

    # create a copy to avoid modifying original dataframe
    df_reduced = df.copy()

    # select only numeric columns
    numeric_df = df_reduced.select_dtypes(include=['number'])

    # calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # ensure target is not dropped
    if target is not None and target in to_drop:
        to_drop.remove(target)

    logger.info(f"Dropping {len(to_drop)} multicollinear features: {to_drop}")

    # drop highly correlated features
    df_reduced = df_reduced.drop(columns=to_drop)

    return df_reduced

def create_feature_sets(df, target=None):
    """
    Create different feature sets for model testing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Complete dataset with all features
    target : str
        Target variable name
    
    Returns:
    --------
    dict
        Dictionary with different feature sets
    """
    logger.info("Creating feature sets")

    feature_sets = {}

    # get numeric columns and exclude id and target
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    if target in numeric_cols:
        numeric_cols.remove(target)

    # original features (from EDA)
    original_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex_Numeric']

    # base features: original features + BMI
    base_features = original_features + ['BMI']

    # intermediate features: base + workout metrics
    intermediate_features = base_features + [
        'Heart_Rate_Reserve',
        'Workout_Volume',
        'Temp_Duration',
        'Temp_Heart_Rate'
    ]

    # advanced features: intermediate + transformations
    # get all log-transformed features
    log_features = [col for col in df.columns if col.startswith('Log_') and col != f'Log_{target}']
    advanced_features = intermediate_features + log_features

    # full feature set (all features except id and target)
    all_features = [col for col in df.columns if col != 'id' and col != target]

    # store feature sets
    feature_sets['original'] = original_features
    feature_sets['base'] = base_features
    feature_sets['intermediate'] = intermediate_features
    feature_sets['advanced'] = advanced_features
    feature_sets['all'] = all_features

    # get selected features using importance
    if target in df.columns:
        _, selected_features = select_features(df, target=target, method='importance', k=15)
        feature_sets['selected'] = selected_features

    logger.info(f"Created {len(feature_sets)} feature sets")

    return feature_sets

def prepare_features(train_df, test_df, original_df=None, combine=True, validate=True):
    """
    Complete pipeline for feature preparation.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    original_df : pandas.DataFrame
        Original dataset (optional)
    combine : bool
        Whether to combine training and original datasets
    validate : bool
        Whether to validate the datasets during processing
    
    Returns:
    --------
    tuple
        Processed training and test datasets, feature sets, validation results
    """
    logger.info("Starting feature preparation pipeline")

    # combine datasets if requested and original_df is provided
    if combine and original_df is not None:
        combined_train = combine_datasets(train_df, original_df)
    else:
        combined_train = train_df.copy()

    # make a copy of the test dataset
    test_copy = test_df.copy()

    # validate raw datasets if requested
    validation_results = {}
    if validate:
        logger.info("Validating raw datasets")
        raw_validation = validate_datasets(combined_train, test_copy, target='Calories')
        validation_results['raw'] = raw_validation

    # create basic features
    train_basic = create_basic_features(combined_train)
    test_basic = create_basic_features(test_copy)

    # create polynomial features for important predictors
    train_poly = create_polynomial_features(train_basic)
    test_poly = create_polynomial_features(test_basic)

    # transform features
    train_transformed = transform_features(train_poly)
    test_transformed = transform_features(test_poly)

    # check multicollinearity
    multicollinear_pairs = check_multicollinearity(train_transformed)

    # create feature sets
    feature_sets = create_feature_sets(train_transformed, target='Calories')

    # validate processed datasets if requested
    if validate:
        logger.info("Validating processed datasets")
        processed_validation = validate_datasets(train_transformed, test_transformed, target='Calories')
        validation_results['processed'] = processed_validation

    logger.info("Feature preparation pipeline completed")

    return train_transformed, test_transformed, feature_sets, multicollinear_pairs, validation_results