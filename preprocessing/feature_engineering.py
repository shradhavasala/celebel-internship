"""
Feature Engineering Module for SmartForecast.
Creates comprehensive time-series features: calendar attributes, lag features,
rolling windows, expanding/exponential moving averages, and target/interaction encodings.
Designed as a stateful transformer to prevent data leakage and support multi-step forecasting.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

import config
from utils.helpers import logger, timer, FeatureEngineeringError


class FeatureEngineer:
    """
    Stateful feature engineering transformer that generates rich time-series features
    and maintains historical group statistics for zero-leakage test/future inference.
    """

    def __init__(
        self,
        lag_days: List[int] = config.LAG_DAYS,
        rolling_windows: List[int] = config.ROLLING_WINDOWS
    ):
        """
        Initialize the FeatureEngineer with specified lag steps and rolling window lengths.

        Args:
            lag_days: List of lag periods (in days).
            rolling_windows: List of window lengths (in days) for rolling statistics.
        """
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        self.store_stats: Dict[int, Dict[str, float]] = {}
        self.item_stats: Dict[int, Dict[str, float]] = {}
        self.store_item_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.global_mean: float = 0.0
        self.feature_names: List[str] = []
        self._is_fitted: bool = False

    @timer
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit encodings and group statistics on the training DataFrame.

        Args:
            df: Training DataFrame containing target column ('sales').

        Returns:
            self: Fitted FeatureEngineer instance.
        """
        if config.TARGET_COL not in df.columns:
            raise FeatureEngineeringError(f"Cannot fit FeatureEngineer: '{config.TARGET_COL}' target missing.")

        logger.info("Fitting FeatureEngineer: computing target encodings and interaction statistics...")
        self.global_mean = float(df[config.TARGET_COL].mean())

        # Store encodings
        store_group = df.groupby(config.STORE_COL)[config.TARGET_COL]
        for store_id, group in store_group:
            self.store_stats[int(store_id)] = {
                "store_target_mean": float(group.mean()),
                "store_target_std": float(group.std(ddof=0)) if len(group) > 1 else 0.0,
                "store_target_max": float(group.max())
            }

        # Item encodings
        item_group = df.groupby(config.ITEM_COL)[config.TARGET_COL]
        for item_id, group in item_group:
            self.item_stats[int(item_id)] = {
                "item_target_mean": float(group.mean()),
                "item_target_std": float(group.std(ddof=0)) if len(group) > 1 else 0.0,
                "item_target_max": float(group.max())
            }

        # Store x Item interaction encodings
        store_item_group = df.groupby([config.STORE_COL, config.ITEM_COL])[config.TARGET_COL]
        for (store_id, item_id), group in store_item_group:
            self.store_item_stats[(int(store_id), int(item_id))] = {
                "store_item_target_mean": float(group.mean()),
                "store_item_target_std": float(group.std(ddof=0)) if len(group) > 1 else 0.0
            }

        self._is_fitted = True
        logger.info(f"FeatureEngineer successfully fitted on {len(self.store_stats)} stores and {len(self.item_stats)} items.")
        return self

    def _create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar and date features from datetime column."""
        dates = pd.to_datetime(df[config.DATE_COL])
        
        df["year"] = dates.dt.year.astype(np.int16)
        df["month"] = dates.dt.month.astype(np.int8)
        df["day"] = dates.dt.day.astype(np.int8)
        df["dayofweek"] = dates.dt.dayofweek.astype(np.int8)
        df["quarter"] = dates.dt.quarter.astype(np.int8)
        
        # ISO week of year
        df["weekofyear"] = dates.dt.isocalendar().week.astype(np.int8)
        df["week"] = df["weekofyear"]  # Alias as requested
        
        # Boolean / binary calendar features
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
        df["weekend"] = df["is_weekend"]  # Alias as requested
        df["is_month_start"] = dates.dt.is_month_start.astype(np.int8)
        df["is_month_end"] = dates.dt.is_month_end.astype(np.int8)

        # Cyclical sine/cosine transformations for month and dayofweek
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
        
        return df

    def _create_lag_and_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lag, rolling window, expanding, and exponential moving average features per group."""
        if config.TARGET_COL not in df.columns:
            return df

        # Group by store and item
        grouped = df.groupby([config.STORE_COL, config.ITEM_COL])[config.TARGET_COL]

        # 1. Lag Features (Lag 1, 7, 14, 30, etc.)
        for lag in self.lag_days:
            df[f"lag_{lag}"] = grouped.shift(lag).astype(np.float32)

        # 2. Rolling Statistics (Mean, Std, Max, Min)
        # Shift by 1 first to prevent target leakage from current day's sales!
        shifted_target = grouped.shift(1)
        for window in self.rolling_windows:
            df[f"rolling_mean_{window}"] = shifted_target.rolling(window=window, min_periods=1).mean().astype(np.float32)
            df[f"rolling_std_{window}"] = shifted_target.rolling(window=window, min_periods=1).std().fillna(0).astype(np.float32)
            df[f"rolling_max_{window}"] = shifted_target.rolling(window=window, min_periods=1).max().astype(np.float32)
            df[f"rolling_min_{window}"] = shifted_target.rolling(window=window, min_periods=1).min().astype(np.float32)

        # 3. Expanding Mean (historical expanding average up to t-1)
        df["expanding_mean"] = shifted_target.expanding(min_periods=1).mean().astype(np.float32)

        # 4. Exponential Moving Average (EMA over multiple spans, shifted by 1)
        for span in [7, 14, 30]:
            df[f"ema_{span}"] = (
                df.groupby([config.STORE_COL, config.ITEM_COL])[config.TARGET_COL]
                .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
                .astype(np.float32)
            )

        return df

    def _apply_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encodings for store, item, and store × item interactions."""
        # Map store encodings
        df["store_target_mean"] = df[config.STORE_COL].map(
            lambda s: self.store_stats.get(int(s), {}).get("store_target_mean", self.global_mean)
        ).astype(np.float32)
        df["store_target_std"] = df[config.STORE_COL].map(
            lambda s: self.store_stats.get(int(s), {}).get("store_target_std", 0.0)
        ).astype(np.float32)

        # Map item encodings
        df["item_target_mean"] = df[config.ITEM_COL].map(
            lambda i: self.item_stats.get(int(i), {}).get("item_target_mean", self.global_mean)
        ).astype(np.float32)
        df["item_target_std"] = df[config.ITEM_COL].map(
            lambda i: self.item_stats.get(int(i), {}).get("item_target_std", 0.0)
        ).astype(np.float32)

        # Store x Item interaction statistics
        df["store_item_target_mean"] = df.apply(
            lambda row: self.store_item_stats.get((int(row[config.STORE_COL]), int(row[config.ITEM_COL])), {}).get("store_item_target_mean", self.global_mean),
            axis=1
        ).astype(np.float32)
        
        # Explicit store × item interaction ratio / product feature
        df["store_item_interaction"] = (df["store_target_mean"] * df["item_target_mean"] / (self.global_mean + 1e-5)).astype(np.float32)
        
        return df

    @timer
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features on any input DataFrame using fitted encodings.

        Args:
            df: Input DataFrame (must be sorted by store, item, date).

        Returns:
            pd.DataFrame: Feature-engineered DataFrame.
        """
        if not self._is_fitted:
            raise FeatureEngineeringError("FeatureEngineer must be fitted before calling transform().")

        logger.info(f"Transforming DataFrame (shape={df.shape}) with feature engineering pipeline...")
        df_feat = df.copy()

        # Ensure datetime and sorting
        if not np.issubdtype(df_feat[config.DATE_COL].dtype, np.datetime64):
            df_feat[config.DATE_COL] = pd.to_datetime(df_feat[config.DATE_COL])
        df_feat.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)
        df_feat.reset_index(drop=True, inplace=True)

        # 1. Calendar features
        df_feat = self._create_calendar_features(df_feat)

        # 2. Lag and Rolling Window features (if target or historical target lags are present)
        df_feat = self._create_lag_and_window_features(df_feat)

        # 3. Target encodings & Interactions
        df_feat = self._apply_encodings(df_feat)

        # Fill any remaining NaNs in lag/rolling features (e.g., initial days of the time series) with group/expanding means
        for col in df_feat.columns:
            if col not in [config.DATE_COL, config.STORE_COL, config.ITEM_COL, config.TARGET_COL]:
                if df_feat[col].isnull().sum() > 0:
                    # Fill within store x item group first, then global fill
                    df_feat[col] = (
                        df_feat.groupby([config.STORE_COL, config.ITEM_COL])[col]
                        .transform(lambda x: x.bfill().ffill())
                    )
                    if df_feat[col].isnull().sum() > 0:
                        df_feat[col] = df_feat[col].fillna(df_feat[col].median()).fillna(0)

        # Register engineered feature names
        excluded_cols = {config.DATE_COL, config.TARGET_COL, "id"}
        self.feature_names = [c for c in df_feat.columns if c not in excluded_cols]

        logger.info(f"Feature engineering complete. Total engineered features: {len(self.feature_names)}")
        return df_feat

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit encodings on df and immediately return transformed features."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """Return the list of generated feature column names."""
        return self.feature_names
