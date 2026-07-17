"""
Visualization Module for SmartForecast.
Generates interactive, rich, state-of-the-art Plotly charts for Exploratory Data Analysis,
model evaluation, residual analysis, feature importance, and multi-step demand forecasts.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from utils.helpers import logger, timer


class ChartBuilder:
    """
    Class providing modular, responsive Plotly chart builders with consistent rich styling.
    """

    @staticmethod
    def _get_base_layout(title: str, height: int = 450) -> Dict[str, Any]:
        """Return standardized layout settings for high-impact aesthetics."""
        return dict(
            title=dict(text=f"<b>{title}</b>", font=dict(size=18, family="Inter, Roboto, sans-serif")),
            template=config.PLOTLY_THEME,
            height=height,
            margin=dict(l=50, r=30, t=60, b=50),
            hoverlabel=dict(
                bgcolor="#0F172A",
                font=dict(size=13, family="Inter, Roboto, sans-serif", color="#FFFFFF"),
                bordercolor="#3B82F6"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

    @staticmethod
    @timer
    def plot_daily_sales_trend(
        df: pd.DataFrame,
        store_id: Optional[int] = None,
        item_id: Optional[int] = None
    ) -> go.Figure:
        """
        Plot interactive daily sales trend line.

        Args:
            df: Sales DataFrame.
            store_id: Optional store filter.
            item_id: Optional item filter.

        Returns:
            go.Figure: Interactive line chart.
        """
        data = df.copy()
        if store_id:
            data = data[data[config.STORE_COL] == store_id]
        if item_id:
            data = data[data[config.ITEM_COL] == item_id]

        daily_agg = data.groupby(config.DATE_COL)[config.TARGET_COL].sum().reset_index()

        fig = px.line(
            daily_agg,
            x=config.DATE_COL,
            y=config.TARGET_COL,
            color_discrete_sequence=[config.COLOR_ACTUAL],
            labels={config.DATE_COL: "Date", config.TARGET_COL: "Total Sales"}
        )
        fig.update_layout(
            **ChartBuilder._get_base_layout("Daily Sales Trend Over Time"),
            hovermode="x unified"
        )
        fig.update_traces(line=dict(width=2))
        return fig

    @staticmethod
    @timer
    def plot_monthly_trend(df: pd.DataFrame) -> go.Figure:
        """Plot monthly aggregated sales volume."""
        data = df.copy()
        data["Year-Month"] = data[config.DATE_COL].dt.to_period("M").astype(str)
        monthly_agg = data.groupby("Year-Month")[config.TARGET_COL].sum().reset_index()

        fig = px.bar(
            monthly_agg,
            x="Year-Month",
            y=config.TARGET_COL,
            color=config.TARGET_COL,
            color_continuous_scale="Blues",
            labels={"Year-Month": "Month", config.TARGET_COL: "Monthly Sales Volume"}
        )
        fig.update_layout(**ChartBuilder._get_base_layout("Monthly Sales Distribution & Seasonality"))
        fig.update_coloraxes(showscale=False)
        return fig

    @staticmethod
    @timer
    def plot_yearly_trend(df: pd.DataFrame) -> go.Figure:
        """Plot yearly comparison box plot or bar trend."""
        data = df.copy()
        data["Year"] = data[config.DATE_COL].dt.year.astype(str)
        yearly_agg = data.groupby("Year")[config.TARGET_COL].sum().reset_index()

        fig = px.bar(
            yearly_agg,
            x="Year",
            y=config.TARGET_COL,
            color="Year",
            color_discrete_sequence=config.COLOR_PALETTE,
            labels={"Year": "Year", config.TARGET_COL: "Total Annual Sales"}
        )
        fig.update_layout(**ChartBuilder._get_base_layout("Year-over-Year Total Demand Comparison"))
        return fig

    @staticmethod
    @timer
    def plot_store_comparison(df: pd.DataFrame) -> go.Figure:
        """Compare total or average sales across stores."""
        store_agg = df.groupby(config.STORE_COL)[config.TARGET_COL].agg(["sum", "mean"]).reset_index()
        store_agg[config.STORE_COL] = "Store " + store_agg[config.STORE_COL].astype(str)
        store_agg.sort_values(by="sum", ascending=False, inplace=True)

        fig = px.bar(
            store_agg,
            x=config.STORE_COL,
            y="sum",
            color="mean",
            color_continuous_scale="Viridis",
            labels={config.STORE_COL: "Store", "sum": "Total Volume", "mean": "Average Daily Demand"}
        )
        fig.update_layout(**ChartBuilder._get_base_layout("Store Performance Comparison (Volume vs Average Demand)"))
        return fig

    @staticmethod
    @timer
    def plot_item_comparison(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """Compare total sales across top items."""
        item_agg = df.groupby(config.ITEM_COL)[config.TARGET_COL].sum().reset_index()
        item_agg[config.ITEM_COL] = "Item " + item_agg[config.ITEM_COL].astype(str)
        item_agg.sort_values(by=config.TARGET_COL, ascending=True, inplace=True)
        if len(item_agg) > top_n:
            item_agg = item_agg.tail(top_n)

        fig = px.bar(
            item_agg,
            x=config.TARGET_COL,
            y=config.ITEM_COL,
            orientation="h",
            color=config.TARGET_COL,
            color_continuous_scale="Teal",
            labels={config.ITEM_COL: "Item", config.TARGET_COL: "Total Historical Demand"}
        )
        fig.update_layout(**ChartBuilder._get_base_layout(f"Top {len(item_agg)} Most In-Demand Items"))
        fig.update_coloraxes(showscale=False)
        return fig

    @staticmethod
    @timer
    def plot_sales_heatmap(df: pd.DataFrame) -> go.Figure:
        """Create a heatmap of Day of Week vs Month seasonality."""
        data = df.copy()
        data["Day of Week"] = data[config.DATE_COL].dt.day_name()
        data["Month"] = data[config.DATE_COL].dt.month_name()

        # Order days and months
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

        pivot = data.pivot_table(
            index="Day of Week",
            columns="Month",
            values=config.TARGET_COL,
            aggfunc="mean"
        )
        pivot = pivot.reindex(index=days_order, columns=months_order)

        fig = px.imshow(
            pivot,
            labels=dict(x="Month", y="Day of Week", color="Avg Sales"),
            color_continuous_scale="YlOrRd",
            aspect="auto"
        )
        fig.update_layout(**ChartBuilder._get_base_layout("Demand Heatmap: Day of Week vs. Month Seasonality"))
        return fig

    @staticmethod
    @timer
    def plot_correlation_matrix(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """Plot interactive correlation matrix of engineered numerical features with sales target."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        excluded = {config.DATE_COL, "id"}
        numeric_cols = [c for c in numeric_cols if c not in excluded]

        if config.TARGET_COL in numeric_cols and len(numeric_cols) > top_n:
            # Pick top features correlated with target
            corrs = df[numeric_cols].corr()[config.TARGET_COL].abs().sort_values(ascending=False)
            selected_cols = corrs.head(top_n).index.tolist()
        else:
            selected_cols = numeric_cols[:top_n]

        corr_matrix = df[selected_cols].corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig.update_layout(**ChartBuilder._get_base_layout("Feature Correlation Matrix (Top Correlated Predictors)"))
        return fig

    @staticmethod
    @timer
    def plot_forecast_vs_actual(
        oof_df: pd.DataFrame,
        store_id: Optional[int] = None,
        item_id: Optional[int] = None
    ) -> go.Figure:
        """
        Overlay Actual vs Predicted historical validation sales.

        Args:
            oof_df: Out-of-fold validation DataFrame with 'date', 'actual', 'predicted'.
            store_id: Optional store filter.
            item_id: Optional item filter.

        Returns:
            go.Figure: Overlay line plot.
        """
        data = oof_df.copy()
        if store_id:
            data = data[data[config.STORE_COL] == store_id]
        if item_id:
            data = data[data[config.ITEM_COL] == item_id]

        daily = data.groupby(config.DATE_COL)[["actual", "predicted"]].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily[config.DATE_COL], y=daily["actual"],
            mode="lines", name="Actual Demand",
            line=dict(color=config.COLOR_ACTUAL, width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=daily[config.DATE_COL], y=daily["predicted"],
            mode="lines", name="Predicted Demand",
            line=dict(color=config.COLOR_PREDICTED, width=2, dash="dash")
        ))
        fig.update_layout(
            **ChartBuilder._get_base_layout("Validation Performance: Ground Truth vs. AI Model Prediction"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    @timer
    def plot_future_forecast(
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        store_id: int = 1,
        item_id: int = 1
    ) -> go.Figure:
        """
        Visualize multi-step future forecast appended to historical sales trend.

        Args:
            historical_df: Historical sales data.
            forecast_df: Future forecasted sales data.
            store_id: Store selection to display.
            item_id: Item selection to display.

        Returns:
            go.Figure: Interactive timeline plot.
        """
        hist_series = historical_df[
            (historical_df[config.STORE_COL] == store_id) & (historical_df[config.ITEM_COL] == item_id)
        ].sort_values(by=config.DATE_COL)

        f_series = forecast_df[
            (forecast_df[config.STORE_COL] == store_id) & (forecast_df[config.ITEM_COL] == item_id)
        ].sort_values(by=config.DATE_COL)

        fig = go.Figure()

        # Historical tail (last 90 days for visual clarity)
        hist_tail = hist_series.tail(90)
        fig.add_trace(go.Scatter(
            x=hist_tail[config.DATE_COL], y=hist_tail[config.TARGET_COL],
            mode="lines+markers", name="Historical Sales",
            line=dict(color=config.COLOR_ACTUAL, width=2.5),
            marker=dict(size=4)
        ))

        # Future forecast
        fig.add_trace(go.Scatter(
            x=f_series[config.DATE_COL], y=f_series["forecast_sales"],
            mode="lines+markers", name="Future Forecast",
            line=dict(color=config.COLOR_FORECAST, width=3),
            marker=dict(size=6, symbol="diamond")
        ))

        # Add connection point
        if not hist_tail.empty and not f_series.empty:
            conn_x = [hist_tail[config.DATE_COL].iloc[-1], f_series[config.DATE_COL].iloc[0]]
            conn_y = [hist_tail[config.TARGET_COL].iloc[-1], f_series["forecast_sales"].iloc[0]]
            fig.add_trace(go.Scatter(
                x=conn_x, y=conn_y,
                mode="lines", showlegend=False,
                line=dict(color=config.COLOR_FORECAST, width=2, dash="dot")
            ))

        fig.update_layout(
            **ChartBuilder._get_base_layout(f"Demand Horizon Forecast — Store {store_id} | Item {item_id}"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    @timer
    def plot_residual_distribution(residuals_df: pd.DataFrame) -> go.Figure:
        """Plot residual error histogram and actual vs residual scatter."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residual Error Distribution (Actual - Predicted)", "Predicted vs Residual Scatter")
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=residuals_df["residual"],
                nbinsx=40,
                marker_color="#8B5CF6",
                name="Residuals"
            ),
            row=1, col=1
        )

        # Scatter
        sample_res = residuals_df.sample(min(1500, len(residuals_df)), random_state=42) if len(residuals_df) > 1500 else residuals_df
        fig.add_trace(
            go.Scatter(
                x=sample_res["predicted"],
                y=sample_res["residual"],
                mode="markers",
                marker=dict(color="#EC4899", size=5, opacity=0.6),
                name="Error Scatter"
            ),
            row=1, col=2
        )

        # Add zero error line to scatter
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        fig.update_layout(**ChartBuilder._get_base_layout("Model Residual & Error Diagnostics", height=420), showlegend=False)
        return fig

    @staticmethod
    @timer
    def plot_feature_importance(feature_importances: Dict[str, float], top_n: int = 15) -> go.Figure:
        """Horizontal bar chart of top N feature importances."""
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_feats.reverse()  # For horizontal bar plot order

        feats = [item[0] for item in sorted_feats]
        scores = [item[1] for item in sorted_feats]

        fig = px.bar(
            x=scores,
            y=feats,
            orientation="h",
            labels={"x": "Importance Score (%)", "y": "Engineered Feature"},
            color=scores,
            color_continuous_scale="Plasma"
        )
        fig.update_layout(**ChartBuilder._get_base_layout(f"Top {len(feats)} Predictive Features Ranking"))
        fig.update_coloraxes(showscale=False)
        return fig
