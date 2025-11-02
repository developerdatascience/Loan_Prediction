import logging
import pandas as pd
from scipy import stats
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_columns_having_nulls_above_threshold(
    data: pd.DataFrame, threshold: float = 50.0
) -> pd.DataFrame:
    """
    Drops columns from the DataFrame that have a percentage of null values above the specified threshold.

    Args:
        data (pd.DataFrame): The input DataFrame.
        threshold (float): The threshold percentage (between 0 and 100). Columns with null percentage above this will be dropped.

    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
    """
    logger.info(f"➡️ Dropping columns with more than {threshold}% null values.")
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100.")

    # Compute percentage of nulls per column and drop columns above the threshold
    null_percent = data.isnull().mean() * 100
    cols_to_drop = null_percent[null_percent > threshold].index.tolist()
    logger.info(f"✅ Dropping columns with null percentage above {threshold}: {cols_to_drop}")
    return data.drop(columns=cols_to_drop).dropna(how='all')

def fill_numeric_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in numeric columns with the mean of the respective columns.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in numeric columns.
    """
    logger.info("✅Filling missing values in numeric columns.")
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)
    return data

def fill_numeric_missing_values_using_interpolation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in numeric columns using linear interpolation.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in numeric columns.
    """
    logger.info("✅ Filling missing values in numeric columns using interpolation.")
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        data[col].interpolate(method='linear', inplace=True)
    return data

def fill_categorical_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in categorical columns with the mode of the respective columns.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in categorical columns.
    """
    logger.info("✅ Filling missing values in categorical columns.")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)
    return data


def encode_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the DataFrame using one-hot encoding.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns encoded.
    """
    logger.info("✅ Encoding categorical columns using one-hot encoding.")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    return pd.get_dummies(data, columns=categorical_cols, drop_first=True)


def drop_columns_with_IDs(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with ID columns dropped
    """
    col_ids = data.columns[data.columns.str.contains('ID', case=False, regex=True)].tolist()
    logger.info(f"Dropping ID columns: {col_ids}")
    return data.drop(columns=col_ids)

def encode_cat_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        mapping = dict(enumerate(df[col].cat.categories))
        logger.info(f"Mapping for column '{col}': {mapping}")

        df[col] = df[col].cat.codes

    return df


def outlier_detection(data: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Detects and removes outliers from numeric columns using the Z-score method.

    Args:
        data (pd.DataFrame): The input DataFrame.
        z_threshold (float): The Z-score threshold to identify outliers.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    logger.info(f"✅ Removing outliers using Z-score method with threshold {z_threshold}.")
    numeric_cols = data.select_dtypes(include=['number']).columns
    z_scores = stats.zscore(data[numeric_cols])
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < z_threshold).all(axis=1)
    return data[filtered_entries]
