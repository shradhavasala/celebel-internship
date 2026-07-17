"""
Streamlit Pages Module for SmartForecast.
Renders main dashboard tabs: Dataset Overview & EDA, Model Training & Evaluation,
Future Horizon Demand Forecasting, and Artifact Downloads.
"""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

import config
from utils.helpers import format_number
from visualization.charts import ChartBuilder


def render_header() -> None:
    """Render professional dashboard header and banner."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1E293B 0%, #3B82F6 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px;">
            <h1 style="margin: 0px; font-size: 32px; font-weight: 700;">🚀 SmartForecast — AI Demand Forecasting System</h1>
            <p style="margin-top: 8px; margin-bottom: 0px; font-size: 15px; opacity: 0.9;">
                Production-grade time-series demand forecasting across multi-store product hierarchies with automated feature engineering, tree-boosting ensembles, and interactive analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_overview_page(df: pd.DataFrame, dataset_info: Dict[str, Any], sidebar_state: Dict[str, Any]) -> None:
    """Render Dataset Overview, KPI metrics, and rich Exploratory Data Analysis charts."""
    st.markdown("### 📋 Dataset Overview & Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Sales Records", format_number(dataset_info.get("num_rows", 0)))
    with col2:
        st.metric("Stores × Items", f"{dataset_info.get('num_stores', 0)} × {dataset_info.get('num_items', 0)}")
    with col3:
        st.metric("Timeline Duration", f"{dataset_info.get('total_days', 0)} days")
    with col4:
        st.metric("Total Historical Volume", format_number(dataset_info.get("total_sales", 0)))
    with col5:
        st.metric("Avg Daily Demand", f"{dataset_info.get('mean_daily_sales', 0):.1f}")

    st.markdown("#### 🔍 Historical Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()
    st.markdown("### 📊 Interactive Exploratory Data Analysis (EDA)")

    eda_tabs = st.tabs([
        "📈 Daily Trends",
        "📅 Monthly & Yearly Seasonality",
        "🏪 Store & Item Performance",
        "🔥 Demand Heatmap",
        "🔗 Correlation Matrix"
    ])

    with eda_tabs[0]:
        st.markdown(f"**Viewing Daily Trend for Store {sidebar_state['selected_store']} | Item {sidebar_state['selected_item']}** (or select all below)")
        show_all = st.checkbox("Show Aggregate Total Across All Stores & Items", value=False)
        if show_all:
            fig_daily = ChartBuilder.plot_daily_sales_trend(df)
        else:
            fig_daily = ChartBuilder.plot_daily_sales_trend(df, store_id=sidebar_state["selected_store"], item_id=sidebar_state["selected_item"])
        st.plotly_chart(fig_daily, use_container_width=True)

    with eda_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ChartBuilder.plot_monthly_trend(df), use_container_width=True)
        with c2:
            st.plotly_chart(ChartBuilder.plot_yearly_trend(df), use_container_width=True)

    with eda_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ChartBuilder.plot_store_comparison(df), use_container_width=True)
        with c2:
            st.plotly_chart(ChartBuilder.plot_item_comparison(df), use_container_width=True)

    with eda_tabs[3]:
        st.plotly_chart(ChartBuilder.plot_sales_heatmap(df), use_container_width=True)

    with eda_tabs[4]:
        st.markdown("Calculates numerical correlations after feature engineering pipeline transformation.")
        if st.session_state.get("engineered_df") is not None:
            st.plotly_chart(ChartBuilder.plot_correlation_matrix(st.session_state["engineered_df"]), use_container_width=True)
        else:
            st.info("ℹ️ Train or transform features first to visualize feature correlation matrix.")


def render_training_page(sidebar_state: Dict[str, Any]) -> None:
    """Render Model Training status, Validation metrics, Residuals, and Feature Importance."""
    st.markdown("### 🧠 Model Training & Cross-Validation Diagnostics")

    if st.session_state.get("trainer") is None or st.session_state["trainer"].final_model is None:
        st.info("👈 Please select an algorithm in the sidebar and click **'Train & Evaluate Model'** to run TimeSeriesSplit training.")
        return

    trainer = st.session_state["trainer"]
    metrics = trainer.cv_metrics
    model_name = trainer.model_name

    # Metric KPI Cards
    st.markdown(f"#### Cross-Validation Performance ({model_name} — {sidebar_state['cv_splits']} Folds)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("RMSE (Root Mean Squared Error)", f"{metrics.get('RMSE', 0.0):.2f}", delta=f"±{metrics.get('RMSE_std', 0.0):.2f}", delta_color="off")
    with m2:
        st.metric("MAE (Mean Absolute Error)", f"{metrics.get('MAE', 0.0):.2f}")
    with m3:
        st.metric("MAPE (Percentage Error)", f"{metrics.get('MAPE', 0.0):.2f}%")
    with m4:
        st.metric("R² Score (Accuracy)", f"{metrics.get('R2', 0.0):.4f}")

    st.divider()

    # Diagnostics Tabs
    diag_tabs = st.tabs([
        "🎯 Ground Truth vs Predicted Overlay",
        "🌟 Top Feature Importances",
        "📉 Residual & Error Analysis"
    ])

    with diag_tabs[0]:
        st.markdown(f"**Validation Fold Overlay for Store {sidebar_state['selected_store']} | Item {sidebar_state['selected_item']}**")
        if trainer.oof_predictions is not None and not trainer.oof_predictions.empty:
            fig_oof = ChartBuilder.plot_forecast_vs_actual(
                trainer.oof_predictions,
                store_id=sidebar_state["selected_store"],
                item_id=sidebar_state["selected_item"]
            )
            st.plotly_chart(fig_oof, use_container_width=True)
        else:
            st.warning("No out-of-fold predictions found.")

    with diag_tabs[1]:
        if trainer.feature_importances:
            fig_imp = ChartBuilder.plot_feature_importance(trainer.feature_importances, top_n=15)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Feature importance not supported by or available from the current model.")

    with diag_tabs[2]:
        if st.session_state.get("residuals_df") is not None:
            fig_res = ChartBuilder.plot_residual_distribution(st.session_state["residuals_df"])
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("Run training to calculate residual distributions.")


def render_forecast_page(df: pd.DataFrame, sidebar_state: Dict[str, Any]) -> None:
    """Render multi-step future horizon demand forecast results."""
    st.markdown("### 🔮 Future Horizon Demand Forecast")

    if st.session_state.get("forecast_df") is None:
        st.info("👈 Click **'Generate Future Forecast'** in the sidebar to run multi-step recursive forecasting across all Store × Item combinations.")
        return

    forecast_df = st.session_state["forecast_df"]
    store_id = sidebar_state["selected_store"]
    item_id = sidebar_state["selected_item"]

    st.markdown(f"#### Interactive Forecast Timeline — Store {store_id} | Item {item_id}")
    fig_future = ChartBuilder.plot_future_forecast(
        historical_df=df,
        forecast_df=forecast_df,
        store_id=store_id,
        item_id=item_id
    )
    st.plotly_chart(fig_future, use_container_width=True)

    st.markdown("#### Forecast Tabular Data Preview")
    filtered_fc = forecast_df[(forecast_df[config.STORE_COL] == store_id) & (forecast_df[config.ITEM_COL] == item_id)]
    st.dataframe(filtered_fc, use_container_width=True)


def render_export_page() -> None:
    """Render artifact download portal for predictions, forecasts, and evaluation reports."""
    st.markdown("### 💾 Export & Download Portal")
    st.markdown("Download generated CSV predictions, multi-step future forecasts, and evaluation reports.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 📄 Kaggle Test Predictions")
        if "test_predictions_df" in st.session_state and st.session_state["test_predictions_df"] is not None:
            csv_data = st.session_state["test_predictions_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download predictions.csv",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("Available after running batch predictions on test.csv.")

    with c2:
        st.markdown("#### 📈 Future Demand Horizon Forecast")
        if "forecast_df" in st.session_state and st.session_state["forecast_df"] is not None:
            fc_data = st.session_state["forecast_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download future_forecast.csv",
                data=fc_data,
                file_name="future_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("Available after generating future forecast.")

    with c3:
        st.markdown("#### 📑 Evaluation Report & Residuals")
        if "eval_report" in st.session_state and st.session_state["eval_report"] is not None:
            report_text = st.session_state["eval_report"].get("report_text", "")
            st.download_button(
                label="📥 Download evaluation_report.txt",
                data=report_text.encode("utf-8"),
                file_name="evaluation_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.caption("Available after training and evaluating a model.")
