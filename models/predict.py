"""
Prediction & Forecasting Engine for SmartForecast.
Handles batch test-set prediction (Kaggle submission format) and multi-step
recursive forecasting into the future while dynamically updating lags and rolling statistics.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

import config
from utils.helpers import logger, timer, InferenceError
from preprocessing.feature_engineering import FeatureEngineer


class ForecastingEngine:
    """
    Class responsible for batch inference on test sets and multi-step recursive
    time-series forecasting into unknown future horizons.
    """

    @staticmethod
    @timer
    def predict_batch(
        model: Any,
        fe: FeatureEngineer,
        test_df: pd.DataFrame,
        save_to_disk: bool = True
    ) -> pd.DataFrame:
        """
        Run inference on a provided test DataFrame (e.g., test.csv) and export predictions.

        Args:
            model: Trained regression model.
            fe: Fitted FeatureEngineer instance.
            test_df: Test DataFrame containing date, store, item (and optional 'id').
            save_to_disk: Whether to export predictions.csv.

        Returns:
            pd.DataFrame: DataFrame containing date, store, item, predicted sales (and id if present).
        """
        if not model or not fe:
            raise InferenceError("Cannot predict: model or FeatureEngineer is missing.")

        logger.info(f"Running batch prediction on {len(test_df):,} test records...")
        
        # Keep original indices and ids if present
        has_id = "id" in test_df.columns
        ids = test_df["id"].values if has_id else None

        # Transform test data using fitted feature engineer
        feat_df = fe.transform(test_df)
        feature_cols = fe.get_feature_names()

        missing_cols = [c for c in feature_cols if c not in feat_df.columns]
        if missing_cols:
            raise InferenceError(f"Missing feature columns in test dataset after transform: {missing_cols[:5]}")

        X_test = feat_df[feature_cols]
        preds = model.predict(X_test)
        preds = np.maximum(np.round(preds), 0).astype(int)

        result_data = {
            config.DATE_COL: feat_df[config.DATE_COL].values,
            config.STORE_COL: feat_df[config.STORE_COL].values,
            config.ITEM_COL: feat_df[config.ITEM_COL].values,
            config.TARGET_COL: preds
        }
        if has_id and ids is not None and len(ids) == len(preds):
            result_data["id"] = ids

        pred_df = pd.DataFrame(result_data)
        
        # If ID is present, format specifically for Kaggle sample_submission format (`id`, `sales`)
        if save_to_disk:
            out_path = config.OUTPUTS_DIR / "predictions.csv"
            pred_df.to_csv(out_path, index=False)
            logger.info(f"Saved batch predictions to disk: {out_path.resolve()}")

        return pred_df

    @staticmethod
    @timer
    def forecast_future(
        model: Any,
        fe: FeatureEngineer,
        historical_df: pd.DataFrame,
        horizon_days: int = config.DEFAULT_FORECAST_HORIZON,
        store_ids: Optional[List[int]] = None,
        item_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Perform multi-step recursive demand forecasting for future N days.
        Dynamically calculates and updates lag and rolling window features for every step.

        Args:
            model: Trained regression model.
            fe: Fitted FeatureEngineer instance.
            historical_df: Historical sales DataFrame containing past ground-truth targets.
            horizon_days: Number of days into the future to forecast.
            store_ids: Optional filter list of specific stores to forecast.
            item_ids: Optional filter list of specific items to forecast.

        Returns:
            pd.DataFrame: Future forecast table containing date, store, item, forecasted sales.
        """
        if not model or not fe:
            raise InferenceError("Cannot run multi-step forecast: model or FeatureEngineer missing.")

        logger.info(f"Starting multi-step recursive future forecast for horizon = {horizon_days} days...")

        if historical_df.empty or config.TARGET_COL not in historical_df.columns:
            raise InferenceError("Historical DataFrame is empty or missing 'sales' target column.")

        # Ensure correct datetimes and types
        hist_clean = historical_df.copy()
        if not np.issubdtype(hist_clean[config.DATE_COL].dtype, np.datetime64):
            hist_clean[config.DATE_COL] = pd.to_datetime(hist_clean[config.DATE_COL])
        hist_clean[config.STORE_COL] = hist_clean[config.STORE_COL].astype(int)
        hist_clean[config.ITEM_COL] = hist_clean[config.ITEM_COL].astype(int)

        # Filter specific stores/items if requested to speed up targeted UI queries
        if store_ids:
            hist_clean = hist_clean[hist_clean[config.STORE_COL].isin(store_ids)]
        if item_ids:
            hist_clean = hist_clean[hist_clean[config.ITEM_COL].isin(item_ids)]

        stores = hist_clean[config.STORE_COL].unique()
        items = hist_clean[config.ITEM_COL].unique()
        max_hist_date = hist_clean[config.DATE_COL].max()

        # To keep recursive transformations fast, we only keep the last 60 days of historical data
        # which is more than enough for lag_30 and rolling_30 windows.
        cutoff_date = max_hist_date - pd.Timedelta(days=65)
        working_df = hist_clean[hist_clean[config.DATE_COL] >= cutoff_date].copy()
        working_df.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)

        feature_cols = fe.get_feature_names()
        if not feature_cols:
            # Run dummy transform to ensure feature_names is populated
            fe.transform(working_df)
            feature_cols = fe.get_feature_names()

        future_records = []

        # Iterate step-by-step through each future day in the horizon
        for step in range(1, horizon_days + 1):
            target_date = max_hist_date + pd.Timedelta(days=step)
            
            # Create placeholder rows for this future date with sales=NaN
            step_rows = []
            for st in stores:
                for it in items:
                    step_rows.append({
                        config.DATE_COL: target_date,
                        config.STORE_COL: st,
                        config.ITEM_COL: it,
                        config.TARGET_COL: np.nan
                    })
            step_df = pd.DataFrame(step_rows)

            # Append to working_df
            combined_df = pd.concat([working_df, step_df], ignore_index=True)
            
            # Run feature engineering on combined tail
            transformed = fe.transform(combined_df)
            
            # Isolate the current target date rows
            current_step_feats = transformed[transformed[config.DATE_COL] == target_date]
            X_future = current_step_feats[feature_cols]

            # Predict
            step_preds = model.predict(X_future)
            step_preds = np.maximum(np.round(step_preds), 0).astype(int)

            # Assign predicted sales back to combined_df so next step uses them as lag/rolling inputs!
            for (st, it, pr) in zip(current_step_feats[config.STORE_COL], current_step_feats[config.ITEM_COL], step_preds):
                mask = (combined_df[config.DATE_COL] == target_date) & (combined_df[config.STORE_COL] == st) & (combined_df[config.ITEM_COL] == it)
                combined_df.loc[mask, config.TARGET_COL] = pr
                future_records.append({
                    config.DATE_COL: target_date,
                    config.STORE_COL: int(st),
                    config.ITEM_COL: int(it),
                    "forecast_sales": int(pr),
                    "step": step
                })

            # Update working_df for next iteration (keeping tail manageable)
            new_cutoff = target_date - pd.Timedelta(days=65)
            working_df = combined_df[combined_df[config.DATE_COL] >= new_cutoff].copy()

        forecast_df = pd.DataFrame(future_records)
        forecast_df.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)
        forecast_df.reset_index(drop=True, inplace=True)

        logger.info(f"Recursive forecast completed successfully. Total forecast records: {len(forecast_df):,}")
        
        # Save to disk
        out_path = config.OUTPUTS_DIR / f"future_forecast_{horizon_days}d.csv"
        forecast_df.to_csv(out_path, index=False)
        logger.info(f"Exported future forecast table to: {out_path.resolve()}")

        return forecast_df
