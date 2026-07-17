"""
Streamlit Sidebar Module for SmartForecast.
Renders data upload controls, algorithm selectors, training triggers,
store/item filtering, and forecast horizon sliders.
"""

from typing import Dict, Any, List
import streamlit as st
import pandas as pd

import config
from models.model_factory import ModelFactory


def render_sidebar(stores: List[int], items: List[int]) -> Dict[str, Any]:
    """
    Render the Streamlit sidebar and return user selection dictionary.

    Args:
        stores: Available store IDs from loaded dataset.
        items: Available item IDs from loaded dataset.

    Returns:
        Dict[str, Any]: Dictionary containing sidebar UI selections and trigger states.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding-bottom: 10px;">
                <h2 style="color: #3B82F6; margin-bottom: 0px;">SmartForecast</h2>
                <p style="color: #64748B; font-size: 13px;">AI-Powered Demand Engine</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

        # 1. Dataset Source & Upload
        st.subheader("1. Data Source")
        data_source = st.radio(
            "Select Data Source:",
            options=["Generate Sample Dataset (Kaggle Demo)", "Upload Custom CSV (train.csv)"],
            index=0,
            help="Choose the built-in synthetic Kaggle sales dataset or upload your own historical CSV."
        )

        uploaded_train = None
        uploaded_test = None
        if data_source == "Upload Custom CSV (train.csv)":
            uploaded_train = st.file_uploader(
                "Upload train.csv (date, store, item, sales):",
                type=["csv"],
                key="train_uploader"
            )
            uploaded_test = st.file_uploader(
                "Upload test.csv (optional for Kaggle predictions):",
                type=["csv"],
                key="test_uploader"
            )

        st.divider()

        # 2. Model Algorithm Selector
        st.subheader("2. Model & Training")
        available_models = ModelFactory.get_available_models()
        selected_model = st.selectbox(
            "Select Regression Algorithm:",
            options=available_models,
            index=0,
            help="LightGBM is the primary high-performance baseline. XGBoost and CatBoost provide competitive tree boosting."
        )

        cv_splits = st.slider(
            "Cross-Validation Folds:",
            min_value=3,
            max_value=10,
            value=config.DEFAULT_CV_SPLITS,
            step=1,
            help="Number of TimeSeriesSplit folds for validation."
        )

        train_button = st.button(
            "🚀 Train & Evaluate Model",
            type="primary",
            use_container_width=True
        )

        st.divider()

        # 3. Forecast Settings & Filters
        st.subheader("3. Forecasting Horizon & Filters")
        forecast_horizon = st.slider(
            "Future Forecast Horizon (Days):",
            min_value=7,
            max_value=90,
            value=config.DEFAULT_FORECAST_HORIZON,
            step=7,
            help="Number of days into the future to predict demand recursively."
        )

        selected_store = st.selectbox(
            "Filter / Inspect Store:",
            options=stores if stores else [1],
            index=0 if stores else 0
        )

        selected_item = st.selectbox(
            "Filter / Inspect Item:",
            options=items if items else [1],
            index=0 if items else 0
        )

        forecast_button = st.button(
            "📈 Generate Future Forecast",
            use_container_width=True
        )

        st.divider()

        # 4. Export & Download Section
        st.subheader("4. Artifact Exports")
        export_mode = st.radio(
            "Select Export Mode:",
            options=["Kaggle Test Predictions (predictions.csv)", "Future Horizon Forecast (CSV)", "Evaluation Summary Report"],
            index=0
        )

        return {
            "data_source": data_source,
            "uploaded_train": uploaded_train,
            "uploaded_test": uploaded_test,
            "selected_model": selected_model,
            "cv_splits": cv_splits,
            "train_button": train_button,
            "forecast_horizon": forecast_horizon,
            "selected_store": selected_store,
            "selected_item": selected_item,
            "forecast_button": forecast_button,
            "export_mode": export_mode
        }
