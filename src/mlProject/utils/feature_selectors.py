import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List
import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.mlProject import logger



def feature_selection(X: pd.DataFrame, y: pd.Series, threshold=0.01) -> List[str]:
    """
    Select important features using RandomForest feature importances.
    
    Args:
        X (pd.DataFrame): input features
        y (pd.Series): target variable
        threshold (float): importance threshold to select features
    Returns:
        List[str]: List of selected important features
    """
    logger.info("✅ Selecting important features using RandomForest feature importances.")

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)

    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_selector.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    
    selected_features = feature_importances[feature_importances['Importance'] >= threshold]['Feature'].tolist()
    logger.info(f"➡️ Selected features: {selected_features}")

    return selected_features


def select_features_by_correlation(
    data: pd.DataFrame, target_column: str, threshold: float = 0.1
) -> List[str]:
    """
    Select features based on correlation with the target variable.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        threshold (float): The correlation threshold. Features with absolute correlation above this will be selected.

    Returns:
        List[str]: List of selected feature names.
    """
    logger.info(f"➡️ Selecting features with correlation above {threshold} with target '{target_column}'.")

    correlations = data.corr()[target_column].abs()
    selected_features = correlations[correlations > threshold].index.tolist()
    selected_features.remove(target_column)  # Remove target column from features

    logger.info(f"✅ Selected features based on correlation: {selected_features}")
    return selected_features


def select_features_by_statistical_test(
    data: pd.DataFrame, target_column: str, alpha: float = 0.05
) -> List[str]:
    """
    Select features based on statistical significance using t-test.

    Args:
        data (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        alpha (float): Significance level for the t-test.

    Returns:
        List[str]: List of selected feature names.
    """
    logger.info(f"➡️ Selecting features using t-test with alpha = {alpha}.")

    selected_features = []
    target_values = data[target_column].unique()

    for column in data.columns:
        if column == target_column:
            continue

        group1 = data[data[target_column] == target_values[0]][column]
        group2 = data[data[target_column] == target_values[1]][column]

        t_stat, p_value = stats.ttest_ind(group1.dropna(), group2.dropna())

        if p_value < alpha:
            selected_features.append(column)

    logger.info(f"✅ Selected features based on statistical test: {selected_features}")
    return selected_features


def select_features_using_VIF(
    data: pd.DataFrame, threshold: float = 5.0
) -> List[str]:
    """
    Select features by removing those with high Variance Inflation Factor (VIF).

    Args:
        data (pd.DataFrame): The input DataFrame.
        threshold (float): VIF threshold. Features with VIF above this will be removed.

    Returns:
        List[str]: List of selected feature names.
    """

    logger.info(f"➡️ Selecting features using VIF with threshold = {threshold}.")

    features = data.columns.tolist()
    while True:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features
        vif_data["VIF"] = [variance_inflation_factor(data[features].values, i) for i in range(len(features))]

        max_vif = vif_data['VIF'].max()
        if max_vif > threshold:
            feature_to_remove = vif_data.loc[vif_data['VIF'] == max_vif, 'Feature'].values[0]
            features.remove(feature_to_remove)
            logger.info(f"Removing feature '{feature_to_remove}' with VIF = {max_vif}.")
        else:
            break

    logger.info(f"✅ Selected features after VIF check: {features}")
    return features