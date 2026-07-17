"""
Data Cleaning Module for SmartForecast.
Provides automated time-series gap filling, missing value imputation,
and outlier detection/treatment to ensure clean inputs for feature engineering.
"""

from typing import Union, List, Dict, Any, Optional
import pandas as pd
import numpy as np

import config
from utils.helpers import logger, timer, DataValidationError


class DataCleaner:
    """
    Class responsible for cleaning time series data: filling missing dates,
    imputing missing values, and clipping extreme demand outliers.
    """

    @staticmethod
    @timer
    def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure every (store, item) time series has a continuous daily frequency without gaps.
        Missing dates are inserted, and missing sales are interpolated or filled with 0.

        Args:
            df: Input sales DataFrame.

        Returns:
            pd.DataFrame: Continuous daily time series DataFrame.
        """
        logger.info("Checking and filling missing calendar dates across all store × item combinations...")
        if df.empty:
            return df

        # Ensure datetime type
        if not np.issubdtype(df[config.DATE_COL].dtype, np.datetime64):
            df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

        stores = df[config.STORE_COL].unique()
        items = df[config.ITEM_COL].unique()
        min_date = df[config.DATE_COL].min()
        max_date = df[config.DATE_COL].max()

        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        expected_rows = len(stores) * len(items) * len(full_date_range)

        if len(df) == expected_rows:
            logger.info("No missing dates detected across time series. Dataset is continuous.")
            return df

        logger.info(f"Filling date gaps: current rows = {len(df):,}, expected continuous rows = {expected_rows:,}")

        # Create multi-index of (store, item, date)
        multi_idx = pd.MultiIndex.from_product(
            [stores, items, full_date_range],
            names=[config.STORE_COL, config.ITEM_COL, config.DATE_COL]
        )

        # Reindex DataFrame against full product multi-index
        df_indexed = df.set_index([config.STORE_COL, config.ITEM_COL, config.DATE_COL])
        df_filled = df_indexed.reindex(multi_idx).reset_index()

        # Handle any non-target columns if present (e.g. ID column in test data)
        for col in df_filled.columns:
            if col not in [config.STORE_COL, config.ITEM_COL, config.DATE_COL, config.TARGET_COL]:
                # Forward fill metadata columns per group if possible
                df_filled[col] = df_filled.groupby([config.STORE_COL, config.ITEM_COL])[col].ffill().bfill()

        # For target column (sales), interpolate or fill with zero
        if config.TARGET_COL in df_filled.columns:
            # First interpolate linearly within group, then fill remaining NaNs with 0
            df_filled[config.TARGET_COL] = (
                df_filled.groupby([config.STORE_COL, config.ITEM_COL])[config.TARGET_COL]
                .transform(lambda x: x.interpolate(method="linear").fillna(0))
            )
            df_filled[config.TARGET_COL] = np.maximum(np.round(df_filled[config.TARGET_COL]), 0).astype(int)

        # Ensure correct types
        df_filled[config.STORE_COL] = df_filled[config.STORE_COL].astype(int)
        df_filled[config.ITEM_COL] = df_filled[config.ITEM_COL].astype(int)
        df_filled.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)
        df_filled.reset_index(drop=True, inplace=True)

        logger.info(f"Successfully filled missing dates. New shape: {df_filled.shape}")
        return df_filled

    @staticmethod
    @timer
    def handle_missing_values(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        Check for any residual NaN or null values and handle them cleanly.

        Args:
            df: Input DataFrame.
            is_test: Whether this is test data (where target is absent/optional).

        Returns:
            pd.DataFrame: Cleaned DataFrame with zero missing values.
        """
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing == 0:
            logger.info("No missing values found in DataFrame.")
            return df

        logger.info(f"Handling {total_missing} missing values across columns:\n{missing_counts[missing_counts > 0]}")

        # Drop rows where essential identifiers or date are missing
        df = df.dropna(subset=[config.DATE_COL, config.STORE_COL, config.ITEM_COL]).copy()

        # Handle target column if present
        if not is_test and config.TARGET_COL in df.columns:
            if df[config.TARGET_COL].isnull().sum() > 0:
                # Impute missing sales using group median or forward fill
                df[config.TARGET_COL] = (
                    df.groupby([config.STORE_COL, config.ITEM_COL])[config.TARGET_COL]
                    .transform(lambda x: x.ffill().bfill().fillna(x.median()).fillna(0))
                )
                df[config.TARGET_COL] = np.maximum(np.round(df[config.TARGET_COL]), 0).astype(int)

        # Fill any other remaining numeric columns with median, object columns with 'Unknown'
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median()).fillna(0)
                else:
                    df[col] = df[col].fillna("Unknown")

        logger.info("All missing values handled successfully.")
        return df

    @staticmethod
    @timer
    def treat_outliers(
        df: pd.DataFrame,
        method: str = "iqr",
        factor: float = 3.5
    ) -> pd.DataFrame:
        """
        Clip extreme demand spikes/outliers per store × item series to prevent distorting lag features and models.
        Uses a conservative threshold (factor=3.5 by default) so genuine peak seasons aren't erased.

        Args:
            df: Input DataFrame containing sales target.
            method: Outlier detection method ('iqr' or 'zscore').
            factor: Multiplier for IQR threshold (e.g. 3.5 * IQR above Q3).

        Returns:
            pd.DataFrame: DataFrame with clipped extreme target values.
        """
        if config.TARGET_COL not in df.columns:
            return df

        logger.info(f"Treating extreme target outliers using method='{method}' (factor={factor})...")
        df_clean = df.copy()

        def clip_group_outliers(series: pd.Series) -> pd.Series:
            if len(series) < 10:
                return series
            if method.lower() == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + factor * iqr
                lower_bound = max(0, q1 - factor * iqr)
            else:  # Z-score
                mean = series.mean()
                std = series.std()
                if std == 0:
                    return series
                upper_bound = mean + factor * std
                lower_bound = max(0, mean - factor * std)
            return series.clip(lower=lower_bound, upper=upper_bound)

        # Apply clipping per store x item group
        df_clean[config.TARGET_COL] = (
            df_clean.groupby([config.STORE_COL, config.ITEM_COL])[config.TARGET_COL]
            .transform(clip_group_outliers)
        )
        df_clean[config.TARGET_COL] = np.round(df_clean[config.TARGET_COL]).astype(int)

        outliers_modified = (df[config.TARGET_COL] != df_clean[config.TARGET_COL]).sum()
        logger.info(f"Clipped {outliers_modified:,} extreme target values across {len(df):,} total rows.")
        return df_clean

    @staticmethod
    @timer
    def prepare_dataset(
        df: pd.DataFrame,
        is_test: bool = False,
        fill_dates: bool = True,
        clip_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Master cleaning pipeline: fill date gaps, handle missing values, and clip outliers.

        Args:
            df: Input raw DataFrame.
            is_test: Whether dataset is test data.
            fill_dates: Whether to fill continuous calendar date gaps.
            clip_outliers: Whether to clip extreme sales outliers (only applies if target exists).

        Returns:
            pd.DataFrame: Cleaned and prepared DataFrame ready for feature engineering.
        """
        logger.info(f"Executing master data cleaning pipeline (is_test={is_test})...")
        clean_df = df.copy()

        if fill_dates:
            clean_df = DataCleaner.fill_missing_dates(clean_df)

        clean_df = DataCleaner.handle_missing_values(clean_df, is_test=is_test)

        if not is_test and clip_outliers and config.TARGET_COL in clean_df.columns:
            clean_df = DataCleaner.treat_outliers(clean_df, method="iqr", factor=3.5)

        logger.info("Master data cleaning pipeline completed successfully.")
        return clean_df
