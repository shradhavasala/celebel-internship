"""
Model Factory Module for SmartForecast.
Implements the Factory Design Pattern to instantiate supported machine learning
regression models (`LightGBM`, `XGBoost`, `CatBoost`, `Random Forest`) with uniform interfaces.
"""

from typing import Dict, Any, Optional, List
import config
from utils.helpers import logger, ModelTrainingError


class ModelFactory:
    """
    Factory class for instantiating machine learning regression models
    with default or customized hyperparameter configurations.
    """

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of all supported algorithm names.

        Returns:
            List[str]: Names of supported models.
        """
        return list(config.MODEL_CONFIGS.keys())

    @staticmethod
    def get_model(
        model_name: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Instantiate and return a regression model by name.

        Args:
            model_name: Name of the algorithm ('LightGBM', 'XGBoost', 'CatBoost', 'Random Forest').
            custom_params: Optional dictionary of hyperparameter overrides.

        Returns:
            Regresssor instance ready for training or tuning.

        Raises:
            ModelTrainingError: If requested model is not supported or library is missing.
        """
        if model_name not in config.MODEL_CONFIGS:
            raise ModelTrainingError(f"Model '{model_name}' is not supported. Available models: {ModelFactory.get_available_models()}")

        # Merge defaults with overrides
        params = config.MODEL_CONFIGS[model_name].copy()
        if custom_params:
            params.update(custom_params)

        logger.info(f"Instantiating model '{model_name}' with parameters: {params}")

        try:
            if model_name == "LightGBM":
                import lightgbm as lgb
                return lgb.LGBMRegressor(**params)
            
            elif model_name == "XGBoost":
                import xgboost as xgb
                return xgb.XGBRegressor(**params)
            
            elif model_name == "CatBoost":
                from catboost import CatBoostRegressor
                return CatBoostRegressor(**params)
            
            elif model_name == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**params)
            
            else:
                raise ModelTrainingError(f"Unhandled model instantiation: '{model_name}'")

        except ImportError as e:
            msg = f"Failed to import library for '{model_name}'. Ensure required package is installed: {e}"
            logger.error(msg)
            raise ModelTrainingError(msg, details=str(e))
        except Exception as e:
            msg = f"Error initializing model '{model_name}': {e}"
            logger.error(msg)
            raise ModelTrainingError(msg, details=str(e))
