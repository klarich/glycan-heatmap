from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


def train_power_transformer(column: pd.Series) -> pd.DataFrame:
    """
    Trains a PowerTransformer on the data in column, then transforms it.
    :param column: column of data to transform
    :return: transformed df
    """
    col_no_na = column.dropna()
    y_train = np.array(col_no_na)
    y_train = np.array(y_train).reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(y_train)
    y_transform = pt.transform(y_train)
    new_col_name = f"{column.name}_transformed"
    transform_df = pd.DataFrame({column.name: col_no_na, new_col_name: y_transform.squeeze()})
    transform_df.reset_index(inplace=True)
    transform_df.drop(columns=[column.name], inplace=True)
    transform_df.rename(columns={"index": "glycan", new_col_name: column.name}, inplace=True)
    return transform_df


def run_power_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the power transformation for every column in df
    :param df: dataframe of columns to apply the power transformation to. A separate transformation will be trained and
    applied to each column
    :return: Dataframe where each column has been transfomed into a normal distribution with mean of zero and unit
    variance
    """
    transformed_dfs = []
    for col in df.columns:
        # Transform glycan binding data into a normal space (see README.md for why this may be necessary)
        series = df[col]
        transformed_dfs.append(train_power_transformer(series))
    transformed_dfs_merged = reduce(
        lambda left, right: pd.merge(left, right, on="glycan", how="outer"), transformed_dfs
    )
    transformed_dfs_merged.set_index("glycan", inplace=True)
    transformed_dfs_merged = transformed_dfs_merged.T
    transformed_dfs_merged.reset_index(inplace=True)
    transformed_dfs_merged.rename(columns={"index": "target_id"}, inplace=True)
    return transformed_dfs_merged
