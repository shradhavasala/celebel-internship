"""
Main Application Entrypoint for SmartForecast.
Run with: streamlit run app.py
Orchestrates data loading, preprocessing, feature engineering, cross-validation
model training, evaluation, forecasting, and interactive dashboard rendering.
"""

import streamlit as st
import pandas as pd
import numpy as np

import config
from utils.helpers import logger, SmartForecastException
from preprocessing.data_loader import DataLoader
from preprocessing.cleaning import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from models.train import ModelTrainer
from models.evaluate import ModelEvaluator
from models.predict import ForecastingEngine
from dashboard.sidebar import render_sidebar
from dashboard.pages import (
    render_header,
    render_overview_page,
    render_training_page,
    render_forecast_page,
    render_export_page
)


def initialize_session_state() -> None:
    """Initialize essential Streamlit session state variables."""
    if "df_train" not in st.session_state:
        st.session_state["df_train"] = None
    if "df_test" not in st.session_state:
        st.session_state["df_test"] = None
    if "dataset_info" not in st.session_state:
        st.session_state["dataset_info"] = {}
    if "fe" not in st.session_state:
        st.session_state["fe"] = None
    if "trainer" not in st.session_state:
        st.session_state["trainer"] = None
    if "residuals_df" not in st.session_state:
        st.session_state["residuals_df"] = None
    if "eval_report" not in st.session_state:
        st.session_state["eval_report"] = None
    if "forecast_df" not in st.session_state:
        st.session_state["forecast_df"] = None
    if "test_predictions_df" not in st.session_state:
        st.session_state["test_predictions_df"] = None


def load_dataset_into_state(data_source: str, uploaded_train: Any, uploaded_test: Any) -> None:
    """Load historical dataset into session state based on user choice."""
    try:
        if data_source == "Generate Sample Dataset (Kaggle Demo)":
            if st.session_state["df_train"] is None:
                with st.spinner("Generating realistic synthetic Kaggle Demand Forecasting dataset..."):
                    train_df, test_df = DataLoader.generate_sample_dataset(
                        num_stores=4, num_items=10, start_date="2021-01-01", end_date="2023-12-31"
                    )
                    st.session_state["df_train"] = train_df
                    st.session_state["df_test"] = test_df
                    st.session_state["dataset_info"] = DataLoader.get_dataset_info(train_df)
        else:
            if uploaded_train is not None:
                with st.spinner("Parsing and validating uploaded train.csv..."):
                    train_df = DataLoader.load_csv(uploaded_train, is_test=False)
                    st.session_state["df_train"] = train_df
                    st.session_state["dataset_info"] = DataLoader.get_dataset_info(train_df)
            if uploaded_test is not None:
                with st.spinner("Parsing and validating uploaded test.csv..."):
                    test_df = DataLoader.load_csv(uploaded_test, is_test=True)
                    st.session_state["df_test"] = test_df
    except SmartForecastException as e:
        st.error(f"❌ Data Error: {e}")
        logger.error(f"Dataset load failure: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {e}")
        logger.error(f"Unexpected data load failure: {e}")


def handle_training_trigger(sidebar_state: Dict[str, Any]) -> None:
    """Handle model training and cross-validation workflow triggered from sidebar."""
    if st.session_state["df_train"] is None:
        st.warning("⚠️ Please load or generate a dataset first.")
        return

    model_name = sidebar_state["selected_model"]
    cv_splits = sidebar_state["cv_splits"]

    try:
        with st.spinner(f"Cleaning data, creating 40+ time-series features, and running {cv_splits}-fold TimeSeriesSplit training for {model_name}..."):
            # 1. Clean Data
            clean_df = DataCleaner.prepare_dataset(st.session_state["df_train"], is_test=False)

            # 2. Feature Engineering
            fe = FeatureEngineer()
            feat_df = fe.fit_transform(clean_df)
            st.session_state["engineered_df"] = feat_df
            st.session_state["fe"] = fe

            # 3. Cross-Validation & Model Training
            trainer = ModelTrainer(model_name=model_name)
            final_model, cv_metrics, oof_df = trainer.train_cv(
                df=feat_df,
                fe=fe,
                n_splits=cv_splits
            )
            st.session_state["trainer"] = trainer

            # 4. Residual Analysis & Evaluation Report
            if oof_df is not None and not oof_df.empty:
                residuals_df = ModelEvaluator.get_residual_analysis(
                    y_true=oof_df["actual"],
                    y_pred=oof_df["predicted"],
                    df=oof_df
                )
                st.session_state["residuals_df"] = residuals_df

                eval_report = ModelEvaluator.generate_evaluation_report(
                    model_name=model_name,
                    metrics=cv_metrics,
                    residuals_df=residuals_df,
                    feature_importances=trainer.feature_importances,
                    save_to_disk=True
                )
                st.session_state["eval_report"] = eval_report

            # 5. Save Artifact
            trainer.save_model()

            # 6. If test dataset exists, run batch prediction
            if st.session_state["df_test"] is not None:
                test_preds = ForecastingEngine.predict_batch(
                    model=final_model,
                    fe=fe,
                    test_df=st.session_state["df_test"],
                    save_to_disk=True
                )
                st.session_state["test_predictions_df"] = test_preds

            st.success(f"✅ Successfully trained & evaluated {model_name}! RMSE: {cv_metrics['RMSE']:.2f} | R²: {cv_metrics['R2']:.4f}")

    except SmartForecastException as e:
        st.error(f"❌ Training Error: {e}")
        logger.error(f"Training exception: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected training error: {e}")
        logger.error(f"Unexpected training exception: {e}")


def handle_forecast_trigger(sidebar_state: Dict[str, Any]) -> None:
    """Handle multi-step future horizon forecasting triggered from sidebar."""
    if st.session_state.get("trainer") is None or st.session_state["trainer"].final_model is None:
        st.warning("⚠️ Please train a model first before generating future horizon forecasts.")
        return

    horizon = sidebar_state["forecast_horizon"]
    trainer = st.session_state["trainer"]

    try:
        with st.spinner(f"Generating recursive {horizon}-day demand forecast across all Store × Item combinations..."):
            clean_df = DataCleaner.prepare_dataset(st.session_state["df_train"], is_test=False)
            forecast_df = ForecastingEngine.forecast_future(
                model=trainer.final_model,
                fe=trainer.fitted_fe,
                historical_df=clean_df,
                horizon_days=horizon
            )
            st.session_state["forecast_df"] = forecast_df
            st.success(f"✅ Generated {horizon}-day future forecast successfully! Check the 'Future Horizon Forecasting' tab.")
    except SmartForecastException as e:
        st.error(f"❌ Forecasting Error: {e}")
        logger.error(f"Forecasting exception: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected forecasting error: {e}")
        logger.error(f"Unexpected forecasting exception: {e}")


def main() -> None:
    """Main Streamlit execution loop."""
    st.set_page_config(
        page_title="SmartForecast | AI Demand Engine",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Determine store/item lists for filters
    stores = []
    items = []
    if st.session_state["df_train"] is not None:
        stores = sorted(st.session_state["df_train"][config.STORE_COL].unique().tolist())
        items = sorted(st.session_state["df_train"][config.ITEM_COL].unique().tolist())

    # Render Sidebar and get user selections
    sidebar_state = render_sidebar(stores=stores, items=items)

    # Handle dataset loading based on radio selection
    load_dataset_into_state(
        sidebar_state["data_source"],
        sidebar_state["uploaded_train"],
        sidebar_state["uploaded_test"]
    )

    # Refresh store/item lists if just loaded
    if st.session_state["df_train"] is not None and not stores:
        stores = sorted(st.session_state["df_train"][config.STORE_COL].unique().tolist())
        items = sorted(st.session_state["df_train"][config.ITEM_COL].unique().tolist())
        if stores and items:
            sidebar_state["selected_store"] = stores[0]
            sidebar_state["selected_item"] = items[0]

    # Handle action button triggers
    if sidebar_state["train_button"]:
        handle_training_trigger(sidebar_state)

    if sidebar_state["forecast_button"]:
        handle_forecast_trigger(sidebar_state)

    # Render Main Dashboard Pages
    render_header()

    if st.session_state["df_train"] is None:
        st.info("ℹ️ Select **'Generate Sample Dataset (Kaggle Demo)'** in the left sidebar or upload your custom CSV (`train.csv`) to begin.")
        return

    main_tabs = st.tabs([
        "📋 Overview & EDA",
        "🧠 Model Training & Evaluation",
        "🔮 Future Horizon Forecasting",
        "💾 Export & Reports"
    ])

    with main_tabs[0]:
        render_overview_page(
            df=st.session_state["df_train"],
            dataset_info=st.session_state["dataset_info"],
            sidebar_state=sidebar_state
        )

    with main_tabs[1]:
        render_training_page(sidebar_state)

    with main_tabs[2]:
        render_forecast_page(
            df=st.session_state["df_train"],
            sidebar_state=sidebar_state
        )

    with main_tabs[3]:
        render_export_page()


if __name__ == "__main__":
    main()
