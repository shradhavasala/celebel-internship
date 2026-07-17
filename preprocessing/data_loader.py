"""
Data Loader Module for SmartForecast.
Handles loading historical sales CSV data, schema validation, type parsing,
and realistic synthetic dataset generation for out-of-the-box evaluation.
"""

import os
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

import config
from utils.helpers import logger, timer, DataValidationError


class DataLoader:
    """
    Class responsible for loading, validating, and generating Demand Forecasting datasets.
    """

    @staticmethod
    @timer
    def load_csv(
        file_path_or_buffer: Union[str, Path, Any],
        is_test: bool = False
    ) -> pd.DataFrame:
        """
        Load historical sales data from a CSV file or buffer (e.g., Streamlit UploadedFile).

        Args:
            file_path_or_buffer: Path to CSV file or file-like object.
            is_test: If True, validates against test schema (without 'sales' target column).

        Returns:
            pd.DataFrame: Validated and date-sorted pandas DataFrame.

        Raises:
            DataValidationError: If required schema columns are missing or date format is invalid.
        """
        try:
            logger.info(f"Loading dataset from: {file_path_or_buffer if isinstance(file_path_or_buffer, (str, Path)) else 'Streamlit Upload Buffer'}...")
            df = pd.read_csv(file_path_or_buffer)
        except Exception as e:
            msg = f"Failed to read CSV file: {e}"
            logger.error(msg)
            raise DataValidationError(msg, details=str(e))

        # Validate Schema
        required_cols = config.REQUIRED_TEST_COLS if is_test else config.REQUIRED_TRAIN_COLS
        DataLoader.validate_schema(df, required_cols)

        # Parse Date column safely
        try:
            df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="mixed")
        except Exception as e:
            msg = f"Failed to parse column '{config.DATE_COL}' to datetime: {e}"
            logger.error(msg)
            raise DataValidationError(msg, details=str(e))

        # Ensure Store and Item columns are clean integers or strings
        df[config.STORE_COL] = df[config.STORE_COL].astype(int)
        df[config.ITEM_COL] = df[config.ITEM_COL].astype(int)

        if not is_test and config.TARGET_COL in df.columns:
            df[config.TARGET_COL] = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
            # Log any missing targets
            missing_targets = df[config.TARGET_COL].isnull().sum()
            if missing_targets > 0:
                logger.warning(f"Found {missing_targets} rows with missing target '{config.TARGET_COL}'.")

        # Sort cleanly by store, item, and date
        df.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Loaded DataFrame with shape: {df.shape} | Stores: {df[config.STORE_COL].nunique()} | Items: {df[config.ITEM_COL].nunique()} | Date Range: {df[config.DATE_COL].min().date()} to {df[config.DATE_COL].max().date()}")
        return df

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_cols: List[str]) -> None:
        """
        Check that all required columns exist in the DataFrame.

        Args:
            df: DataFrame to check.
            required_cols: List of mandatory column names.

        Raises:
            DataValidationError: If any column is missing.
        """
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"Dataset is missing required columns: {missing_cols}. Required: {required_cols}"
            logger.error(msg)
            raise DataValidationError(msg, details={"missing_columns": missing_cols})

    @staticmethod
    @timer
    def generate_sample_dataset(
        num_stores: int = 4,
        num_items: int = 10,
        start_date: str = "2021-01-01",
        end_date: str = "2023-12-31",
        save_to_disk: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a realistic synthetic Kaggle Demand Forecasting dataset.
        Includes trend, yearly seasonality, weekly seasonality (weekend spikes), and random noise.

        Args:
            num_stores: Number of distinct stores.
            num_items: Number of distinct items per store.
            start_date: Start date of historical sales.
            end_date: End date of historical sales.
            save_to_disk: Whether to save `train.csv` and `test.csv` into `data/` folder.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) where test_df is the final 30 days.
        """
        logger.info(f"Generating realistic sample dataset ({num_stores} stores × {num_items} items, {start_date} to {end_date})...")
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Create base data rows
        records = []
        np.random.seed(config.DEFAULT_RANDOM_STATE)

        # Store base sales multipliers
        store_multipliers = {s: np.random.uniform(0.8, 1.5) for s in range(1, num_stores + 1)}
        # Item base volume
        item_base_sales = {i: np.random.uniform(15, 80) for i in range(1, num_items + 1)}

        # Iterate and generate time series with seasonality
        for store_id in range(1, num_stores + 1):
            for item_id in range(1, num_items + 1):
                base = item_base_sales[item_id] * store_multipliers[store_id]
                
                # Yearly seasonality (peaks in summer and late December)
                day_of_year = date_range.dayofyear
                yearly_season = 0.25 * np.sin(2 * np.pi * day_of_year / 365.25) + 0.15 * np.cos(4 * np.pi * day_of_year / 365.25)
                
                # Weekly seasonality (weekend spike on Friday, Saturday, Sunday)
                day_of_week = date_range.dayofweek
                weekly_season = np.where(day_of_week >= 4, 0.35, -0.10)
                
                # Slight upward trend over time
                trend = np.linspace(0, 0.20, len(date_range))
                
                # Random noise
                noise = np.random.normal(0, 0.12, len(date_range))
                
                # Calculate sales
                sales = base * (1 + yearly_season + weekly_season + trend + noise)
                sales = np.maximum(np.round(sales), 0).astype(int)

                for dt, s in zip(date_range, sales):
                    records.append({
                        config.DATE_COL: dt,
                        config.STORE_COL: store_id,
                        config.ITEM_COL: item_id,
                        config.TARGET_COL: s
                    })

        full_df = pd.DataFrame(records)
        full_df.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # Split into Train (all dates except last 30 days) and Test (last 30 days)
        cutoff_date = full_df[config.DATE_COL].max() - pd.Timedelta(days=config.DEFAULT_FORECAST_HORIZON)
        train_df = full_df[full_df[config.DATE_COL] <= cutoff_date].copy()
        test_df = full_df[full_df[config.DATE_COL] > cutoff_date].copy()

        if save_to_disk:
            train_path = config.DATA_DIR / "train.csv"
            test_path = config.DATA_DIR / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.drop(columns=[config.TARGET_COL]).to_csv(test_path, index=False)
            logger.info(f"Saved synthetic datasets to disk: {train_path.resolve()} and {test_path.resolve()}")

        return train_df, test_df

    @staticmethod
    def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract summary statistics from a loaded DataFrame for dashboard display.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dict[str, Any]: Summary dictionary containing counts and date ranges.
        """
        info = {
            "num_rows": len(df),
            "num_stores": df[config.STORE_COL].nunique() if config.STORE_COL in df.columns else 0,
            "num_items": df[config.ITEM_COL].nunique() if config.ITEM_COL in df.columns else 0,
            "num_series": (df[config.STORE_COL].nunique() * df[config.ITEM_COL].nunique()) if (config.STORE_COL in df.columns and config.ITEM_COL in df.columns) else 0,
            "min_date": df[config.DATE_COL].min().strftime("%Y-%m-%d") if config.DATE_COL in df.columns else "N/A",
            "max_date": df[config.DATE_COL].max().strftime("%Y-%m-%d") if config.DATE_COL in df.columns else "N/A",
            "total_days": (df[config.DATE_COL].max() - df[config.DATE_COL].min()).days + 1 if config.DATE_COL in df.columns else 0,
            "missing_values": int(df.isnull().sum().sum())
        }
        if config.TARGET_COL in df.columns:
            info["total_sales"] = int(df[config.TARGET_COL].sum())
            info["mean_daily_sales"] = float(df[config.TARGET_COL].mean())
            info["median_daily_sales"] = float(df[config.TARGET_COL].median())
            info["max_daily_sales"] = int(df[config.TARGET_COL].max())
            info["min_daily_sales"] = int(df[config.TARGET_COL].min())
        return info
