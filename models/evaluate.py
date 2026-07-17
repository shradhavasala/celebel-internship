"""
Model Evaluation Module for SmartForecast.
Computes comprehensive evaluation metrics (RMSE, MAE, MAPE, R²),
residual error diagnostics, and exports detailed evaluation reports.
"""

from typing import Dict, Any, Union, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from utils.helpers import logger, timer, format_number


class ModelEvaluator:
    """
    Class responsible for calculating forecasting metrics, analyzing errors/residuals,
    and generating human-readable evaluation summaries and export files.
    """

    @staticmethod
    def calculate_metrics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute RMSE, MAE, MAPE, and R² score between ground truth and predictions.

        Args:
            y_true: Actual demand target values.
            y_pred: Predicted demand values.

        Returns:
            Dict[str, float]: Dictionary of metric scores.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.maximum(np.asarray(y_pred, dtype=float), 0)  # Demand cannot be negative

        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))

        # Safe MAPE calculation avoiding division by zero
        non_zero_mask = y_t != 0
        if non_zero_mask.any():
            mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
        else:
            mape = 0.0

        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2
        }

    @staticmethod
    @timer
    def get_residual_analysis(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate a detailed DataFrame containing predictions, residuals, and percentage errors.

        Args:
            y_true: Actual demand target values.
            y_pred: Predicted demand values.
            df: Optional original DataFrame containing date, store, item metadata.

        Returns:
            pd.DataFrame: Residual analysis table.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.maximum(np.asarray(y_pred, dtype=float), 0)

        residuals = y_t - y_p
        abs_residuals = np.abs(residuals)

        # Percentage error where y_true != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_errors = np.where(y_t != 0, (residuals / y_t) * 100.0, 0.0)

        data = {
            "actual": y_t,
            "predicted": y_p,
            "residual": residuals,
            "abs_residual": abs_residuals,
            "percentage_error": pct_errors
        }

        if df is not None:
            if config.DATE_COL in df.columns:
                data[config.DATE_COL] = df[config.DATE_COL].values
            if config.STORE_COL in df.columns:
                data[config.STORE_COL] = df[config.STORE_COL].values
            if config.ITEM_COL in df.columns:
                data[config.ITEM_COL] = df[config.ITEM_COL].values

        res_df = pd.DataFrame(data)
        if config.DATE_COL in res_df.columns and config.STORE_COL in res_df.columns:
            res_df.sort_values(by=[config.STORE_COL, config.ITEM_COL, config.DATE_COL], inplace=True)
            res_df.reset_index(drop=True, inplace=True)

        return res_df

    @staticmethod
    @timer
    def generate_evaluation_report(
        model_name: str,
        metrics: Dict[str, float],
        residuals_df: pd.DataFrame,
        feature_importances: Dict[str, float],
        save_to_disk: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a structured evaluation summary report and save text and CSV outputs.

        Args:
            model_name: Name of evaluated model.
            metrics: Dictionary of evaluated metrics (RMSE, MAE, MAPE, R2).
            residuals_df: Residuals analysis table.
            feature_importances: Dictionary ranking top feature importances.
            save_to_disk: Whether to export evaluation_report.txt and residuals.csv.

        Returns:
            Dict[str, Any]: Summary dictionary for UI display.
        """
        logger.info(f"Generating evaluation report for model '{model_name}'...")

        mean_residual = float(residuals_df["residual"].mean())
        std_residual = float(residuals_df["residual"].std())
        max_underprediction = float(residuals_df["residual"].max())  # actual >>> predicted
        max_overprediction = float(residuals_df["residual"].min())   # actual <<< predicted

        top_5_feats = list(feature_importances.items())[:5]

        report_lines = [
            f"==================================================",
            f"         SMARTFORECAST EVALUATION REPORT         ",
            f"==================================================",
            f"Model Evaluated: {model_name}",
            f"Total Evaluated Records: {len(residuals_df):,}",
            f"",
            f"--------------------------------------------------",
            f"1. CORE ACCURACY METRICS",
            f"--------------------------------------------------",
            f"• RMSE (Root Mean Squared Error): {metrics.get('RMSE', 0.0):.4f}",
            f"• MAE  (Mean Absolute Error)    : {metrics.get('MAE', 0.0):.4f}",
            f"• MAPE (Mean Abs Percent Error) : {metrics.get('MAPE', 0.0):.2f}%",
            f"• R²   (Coefficient of Determ.) : {metrics.get('R2', 0.0):.4f}",
            f"",
            f"--------------------------------------------------",
            f"2. RESIDUAL & ERROR DIAGNOSTICS",
            f"--------------------------------------------------",
            f"• Mean Residual Bias : {mean_residual:+.4f} (Pos = Underpredicts, Neg = Overpredicts)",
            f"• Residual Std Dev   : {std_residual:.4f}",
            f"• Max Underprediction: {max_underprediction:+.2f}",
            f"• Max Overprediction : {max_overprediction:+.2f}",
            f"",
            f"--------------------------------------------------",
            f"3. TOP FEATURE IMPORTANCES",
            f"--------------------------------------------------"
        ]

        for idx, (feat, score) in enumerate(top_5_feats, start=1):
            report_lines.append(f"  {idx}. {feat:<25}: {score:.2f}%")

        report_lines.append("==================================================")
        report_text = "\n".join(report_lines)

        if save_to_disk:
            report_path = config.OUTPUTS_DIR / f"{model_name.lower().replace(' ', '_')}_evaluation_report.txt"
            residuals_path = config.OUTPUTS_DIR / f"{model_name.lower().replace(' ', '_')}_residuals.csv"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            
            residuals_df.to_csv(residuals_path, index=False)
            logger.info(f"Exported evaluation report: {report_path.resolve()} | residuals CSV: {residuals_path.resolve()}")

        return {
            "model_name": model_name,
            "metrics": metrics,
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "max_underprediction": max_underprediction,
            "max_overprediction": max_overprediction,
            "report_text": report_text
        }
