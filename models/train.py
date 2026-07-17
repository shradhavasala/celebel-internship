"""
Model Training Module for SmartForecast.
Implements TimeSeriesSplit cross-validation, validation tracking, early stopping,
feature importance tracking, and model/scaler persistence via joblib.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from utils.helpers import logger, timer, ModelTrainingError
from models.model_factory import ModelFactory
from preprocessing.feature_engineering import FeatureEngineer


class ModelTrainer:
    """
    Class responsible for training regression models using TimeSeriesSplit cross-validation,
    evaluating fold performance, applying early stopping, and persisting artifacts.
    """

    def __init__(
        self,
        model_name: str = "LightGBM",
        custom_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model_name: Name of the model algorithm ('LightGBM', 'XGBoost', 'CatBoost', 'Random Forest').
            custom_params: Optional hyperparameter overrides.
        """
        self.model_name = model_name
        self.custom_params = custom_params or {}
        self.final_model: Optional[Any] = None
        self.fitted_fe: Optional[FeatureEngineer] = None
        self.cv_metrics: Dict[str, float] = {}
        self.feature_importances: Dict[str, float] = {}
        self.oof_predictions: Optional[pd.DataFrame] = None

    @timer
    def train_cv(
        self,
        df: pd.DataFrame,
        fe: FeatureEngineer,
        n_splits: int = config.DEFAULT_CV_SPLITS,
        early_stopping_rounds: int = config.DEFAULT_EARLY_STOPPING_ROUNDS
    ) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
        """
        Run TimeSeriesSplit cross-validation, evaluate metrics, and fit the master final model.

        Args:
            df: Feature-engineered training DataFrame.
            fe: Fitted FeatureEngineer instance.
            n_splits: Number of TimeSeriesSplit validation folds.
            early_stopping_rounds: Number of rounds without validation improvement before early stopping.

        Returns:
            Tuple[Any, Dict[str, float], pd.DataFrame]: (final_model, average_cv_metrics, oof_predictions_df)
        """
        feature_cols = fe.get_feature_names()
        if not feature_cols:
            raise ModelTrainingError("No feature names found in FeatureEngineer. Did you run transform()?")
        
        # Verify columns exist
        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats:
            raise ModelTrainingError(f"Missing feature columns in training DataFrame: {missing_feats[:5]}...")

        # Ensure sorted by date for TimeSeriesSplit across all series, or sort chronologically
        df_sorted = df.sort_values(by=[config.DATE_COL, config.STORE_COL, config.ITEM_COL]).reset_index(drop=True)
        X = df_sorted[feature_cols]
        y = df_sorted[config.TARGET_COL]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_rmses: List[float] = []
        fold_maes: List[float] = []
        fold_mapes: List[float] = []
        fold_r2s: List[float] = []
        oof_records: List[Dict[str, Any]] = []
        best_iterations: List[int] = []

        logger.info(f"Starting {n_splits}-fold TimeSeriesSplit cross-validation for model '{self.model_name}'...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            fold_model = ModelFactory.get_model(self.model_name, self.custom_params)

            # Fit with Early Stopping based on model family
            try:
                if self.model_name == "LightGBM":
                    import lightgbm as lgb
                    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                    fold_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="rmse",
                        callbacks=callbacks
                    )
                    if hasattr(fold_model, "best_iteration_") and fold_model.best_iteration_ > 0:
                        best_iterations.append(fold_model.best_iteration_)
                
                elif self.model_name == "XGBoost":
                    # For XGBoost 2.0+, early_stopping_rounds can be passed to constructor or fit
                    fold_model.set_params(early_stopping_rounds=early_stopping_rounds)
                    fold_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    if hasattr(fold_model, "best_iteration") and fold_model.best_iteration > 0:
                        best_iterations.append(fold_model.best_iteration)
                
                elif self.model_name == "CatBoost":
                    fold_model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                    if hasattr(fold_model, "get_best_iteration") and fold_model.get_best_iteration() is not None:
                        best_iterations.append(fold_model.get_best_iteration())
                
                else:  # Random Forest or others
                    fold_model.fit(X_train, y_train)

            except Exception as e:
                logger.warning(f"Early stopping fit failed on Fold {fold} ({e}), falling back to standard fit...")
                fold_model.fit(X_train, y_train)

            # Predict on validation fold
            preds = fold_model.predict(X_val)
            preds = np.maximum(preds, 0)  # Demand cannot be negative

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            # MAPE avoiding division by zero
            mask = y_val != 0
            mape = float(np.mean(np.abs((y_val[mask] - preds[mask]) / y_val[mask])) * 100) if mask.any() else 0.0
            r2 = r2_score(y_val, preds)

            fold_rmses.append(rmse)
            fold_maes.append(mae)
            fold_mapes.append(mape)
            fold_r2s.append(r2)

            logger.info(f"Fold {fold}/{n_splits} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

            # Store OOF predictions
            val_dates = df_sorted.iloc[val_idx][config.DATE_COL].values
            val_stores = df_sorted.iloc[val_idx][config.STORE_COL].values
            val_items = df_sorted.iloc[val_idx][config.ITEM_COL].values
            for dt, st, it, act, pr in zip(val_dates, val_stores, val_items, y_val, preds):
                oof_records.append({
                    config.DATE_COL: dt,
                    config.STORE_COL: st,
                    config.ITEM_COL: it,
                    "actual": act,
                    "predicted": pr,
                    "fold": fold
                })

        # Calculate average CV metrics
        self.cv_metrics = {
            "RMSE": float(np.mean(fold_rmses)),
            "MAE": float(np.mean(fold_maes)),
            "MAPE": float(np.mean(fold_mapes)),
            "R2": float(np.mean(fold_r2s)),
            "RMSE_std": float(np.std(fold_rmses)),
            "R2_std": float(np.std(fold_r2s))
        }
        self.oof_predictions = pd.DataFrame(oof_records)

        logger.info(f"CV Summary ({n_splits} folds): Avg RMSE = {self.cv_metrics['RMSE']:.2f} | Avg R² = {self.cv_metrics['R2']:.4f}")

        # Train Master Final Model on Full Training Set
        logger.info("Training master final model on 100% of available historical data...")
        master_params = self.custom_params.copy()
        
        # Adjust n_estimators if early stopping found an optimal tree count across folds
        if best_iterations and self.model_name in ["LightGBM", "XGBoost", "CatBoost"]:
            avg_best_iter = int(np.mean(best_iterations))
            if avg_best_iter > 10:
                logger.info(f"Setting final model iterations/trees to {avg_best_iter} based on early stopping CV average.")
                if self.model_name in ["LightGBM", "XGBoost"]:
                    master_params["n_estimators"] = avg_best_iter
                elif self.model_name == "CatBoost":
                    master_params["iterations"] = avg_best_iter

        self.final_model = ModelFactory.get_model(self.model_name, master_params)
        self.final_model.fit(X, y)
        self.fitted_fe = fe

        # Extract Feature Importances
        self._extract_feature_importances(feature_cols)

        return self.final_model, self.cv_metrics, self.oof_predictions

    def _extract_feature_importances(self, feature_cols: List[str]) -> None:
        """Extract and normalize feature importance scores from the final model."""
        if not self.final_model:
            return

        importances = None
        if hasattr(self.final_model, "feature_importances_"):
            importances = self.final_model.feature_importances_
        elif hasattr(self.final_model, "get_feature_importance"):
            importances = self.final_model.get_feature_importance()

        if importances is not None and len(importances) == len(feature_cols):
            # Normalize to percentage
            total = np.sum(importances)
            if total > 0:
                norm_importances = (importances / total) * 100.0
            else:
                norm_importances = importances
            self.feature_importances = {
                col: float(imp) for col, imp in sorted(
                    zip(feature_cols, norm_importances), key=lambda x: x[1], reverse=True
                )
            }
            logger.info(f"Top 5 most important features for {self.model_name}: {list(self.feature_importances.items())[:5]}")

    @timer
    def save_model(self, file_name: Optional[str] = None) -> Path:
        """
        Save the trained master model and its fitted FeatureEngineer to disk via joblib.

        Args:
            file_name: Custom file name (defaults to `{model_name}_smartforecast.joblib`).

        Returns:
            Path: Absolute path to the saved model file.
        """
        if not self.final_model or not self.fitted_fe:
            raise ModelTrainingError("Cannot save model: no trained model or fitted FeatureEngineer found.")

        if not file_name:
            clean_name = self.model_name.lower().replace(" ", "_")
            file_name = f"{clean_name}_smartforecast.joblib"

        file_path = config.MODELS_DIR / file_name
        artifact = {
            "model_name": self.model_name,
            "model": self.final_model,
            "feature_engineer": self.fitted_fe,
            "cv_metrics": self.cv_metrics,
            "feature_importances": self.feature_importances
        }

        joblib.dump(artifact, file_path)
        logger.info(f"Saved complete model artifact ({self.model_name}) to: {file_path.resolve()}")
        return file_path

    @staticmethod
    @timer
    def load_model(file_path: Union[str, Path]) -> Tuple[Any, FeatureEngineer, Dict[str, Any]]:
        """
        Load a saved model artifact from disk.

        Args:
            file_path: Path to `.joblib` model artifact.

        Returns:
            Tuple[Any, FeatureEngineer, Dict[str, Any]]: (model, feature_engineer, metadata_dict)
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise ModelTrainingError(f"Model artifact file not found at: {path_obj.resolve()}")

        logger.info(f"Loading model artifact from: {path_obj.resolve()}...")
        artifact = joblib.load(path_obj)

        model = artifact["model"]
        fe = artifact["feature_engineer"]
        metadata = {
            "model_name": artifact.get("model_name", "Unknown"),
            "cv_metrics": artifact.get("cv_metrics", {}),
            "feature_importances": artifact.get("feature_importances", {})
        }

        logger.info(f"Loaded model '{metadata['model_name']}' successfully.")
        return model, fe, metadata
