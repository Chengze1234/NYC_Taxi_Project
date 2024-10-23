# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:52:35 2024

@author: 24375
"""

# data_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split

def raw_taxi_df(filename: str) -> pd.DataFrame:
    """Load raw taxi dataframe from parquet"""
    return pd.read_parquet(path=filename)

def clean_taxi_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean taxi data by removing NaNs, trips longer than 100 miles, and calculating travel time"""
    clean_df = raw_df.dropna()
    clean_df = clean_df[clean_df["trip_distance"] < 100]
    clean_df["time_deltas"] = clean_df["tpep_dropoff_datetime"] - clean_df["tpep_pickup_datetime"]
    clean_df["time_mins"] = pd.to_numeric(clean_df["time_deltas"]) / (10**9 * 60)  # convert to minutes
    return clean_df

def split_taxi_data(clean_df: pd.DataFrame, x_columns: list[str], y_column: str, train_size: int) -> tuple:
    """Split data into train and test sets"""
    return train_test_split(clean_df[x_columns], clean_df[[y_column]], train_size=train_size)
