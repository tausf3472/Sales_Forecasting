"""
Visualization Module for Sales Forecasting
Creates plots for data exploration, model comparison, and forecast results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, Optional
import os


# Style configuration
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


class SalesVisualizer:
    """
    Creates and saves visualizations for the sales forecasting pipeline.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Parameters
        ----------
        output_dir : str
            Directory to save plot images.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_sales_overview(self, df: pd.DataFrame) -> str:
        """
        Plot historical sales data with trend line and key annotations.

        Returns path to saved figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Sales Data Overview", fontsize=16, fontweight="bold")

        dates = pd.to_datetime(df["date"])

        # --- Top Left: Daily sales with trend ---
        ax = axes[0, 0]
        ax.plot(dates, df["sales"], alpha=0.5, linewidth=0.8, color="steelblue", label="Daily Sales")
        # Rolling average
        rolling_30 = df["sales"].rolling(30, min_periods=1).mean()
        ax.plot(dates, rolling_30, color="red", linewidth=2, label="30-Day Moving Avg")
        ax.set_title("Daily Sales & Trend")
        ax.set_ylabel("Sales")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

        # --- Top Right: Sales distribution ---
        ax = axes[0, 1]
        ax.hist(df["sales"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(df["sales"].mean(), color="red", linestyle="--", label=f"Mean: {df['sales'].mean():.0f}")
        ax.axvline(df["sales"].median(), color="orange", linestyle="--", label=f"Median: {df['sales'].median():.0f}")
        ax.set_title("Sales Distribution")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Frequency")
        ax.legend()

        # --- Bottom Left: Weekly pattern ---
        ax = axes[1, 0]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly = df.groupby("day_of_week")["sales"].mean()
        bars = ax.bar(day_names, weekly.values, color=sns.color_palette("deep", 7), edgecolor="white")
        ax.set_title("Average Sales by Day of Week")
        ax.set_ylabel("Average Sales")
        for bar, val in zip(bars, weekly.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

        # --- Bottom Right: Monthly pattern ---
        ax = axes[1, 1]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = df.groupby("month")["sales"].mean()
        bars = ax.bar(month_names, monthly.values, color=sns.color_palette("coolwarm", 12), edgecolor="white")
        ax.set_title("Average Sales by Month")
        ax.set_ylabel("Average Sales")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "sales_overview.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_sales_decomposition(self, df: pd.DataFrame) -> str:
        """
        Decompose sales into trend, seasonality, and residual components.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle("Sales Time Series Decomposition", fontsize=16, fontweight="bold")

        dates = pd.to_datetime(df["date"])
        sales = df["sales"].values

        # Original
        axes[0].plot(dates, sales, color="steelblue", alpha=0.7, linewidth=0.8)
        axes[0].set_ylabel("Original")
        axes[0].set_title("Observed Sales")

        # Trend (30-day rolling mean)
        trend = pd.Series(sales).rolling(30, center=True, min_periods=1).mean().values
        axes[1].plot(dates, trend, color="red", linewidth=2)
        axes[1].set_ylabel("Trend")
        axes[1].set_title("Trend Component (30-Day Moving Average)")

        # Seasonality (7-day rolling mean of detrended data)
        detrended = sales - trend
        seasonal = pd.Series(detrended).rolling(7, center=True, min_periods=1).mean().values
        axes[2].plot(dates, seasonal, color="green", linewidth=1)
        axes[2].set_ylabel("Seasonality")
        axes[2].set_title("Seasonal Component")

        # Residual
        residual = sales - trend - seasonal
        axes[3].scatter(dates, residual, color="gray", alpha=0.3, s=5)
        axes[3].axhline(0, color="black", linewidth=0.5)
        axes[3].set_ylabel("Residual")
        axes[3].set_title("Residual Component")
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        for ax in axes:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "sales_decomposition.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Bar chart comparing model performance metrics.
        """
        df_results = pd.DataFrame(results).T
        df_results.index.name = "Model"

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

        metrics = ["MAE", "RMSE", "MAPE", "R2"]
        colors = sns.color_palette("viridis", len(df_results))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = df_results[metric].values
            model_names = df_results.index.tolist()

            bars = ax.barh(model_names, values, color=colors, edgecolor="white")
            ax.set_title(metric)
            ax.set_xlabel(metric)

            # Annotate bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}" if metric != "R2" else f"{val:.4f}",
                    va="center",
                    fontsize=9,
                )

            # Highlight best
            if metric == "R2":
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor("red")
            bars[best_idx].set_linewidth(2)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_actual_vs_predicted(
        self,
        y_test: np.ndarray,
        predictions: Dict[str, np.ndarray],
        dates_test: Optional[np.ndarray] = None,
    ) -> str:
        """
        Plot actual vs predicted sales for each model on the test set.
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(
            n_models, 1, figsize=(16, 4 * n_models), sharex=True
        )
        fig.suptitle("Actual vs Predicted Sales (Test Set)", fontsize=16, fontweight="bold")

        if n_models == 1:
            axes = [axes]

        if dates_test is not None:
            x_axis = pd.to_datetime(dates_test)
        else:
            x_axis = np.arange(len(y_test))

        colors = sns.color_palette("tab10", n_models)

        for idx, (name, y_pred) in enumerate(predictions.items()):
            ax = axes[idx]
            # Align x_axis and y_test to match prediction length
            # (LSTM produces fewer predictions due to sequence windowing)
            offset = len(y_test) - len(y_pred)
            x_axis_aligned = x_axis[offset:]
            y_test_aligned = y_test[offset:]

            ax.plot(x_axis_aligned, y_test_aligned, color="black", alpha=0.6, linewidth=1, label="Actual")
            ax.plot(x_axis_aligned, y_pred, color=colors[idx], alpha=0.8, linewidth=1.2, label=f"{name}")
            ax.fill_between(
                x_axis_aligned, y_test_aligned, y_pred, alpha=0.15, color=colors[idx]
            )
            ax.set_ylabel("Sales")
            ax.set_title(f"{name}")
            ax.legend(loc="upper right")

            if dates_test is not None:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "actual_vs_predicted.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_feature_importance(
        self, feature_importances: pd.DataFrame, top_n: int = 20
    ) -> str:
        """
        Horizontal bar chart of top feature importances.
        """
        top_features = feature_importances.head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1],
            color=sns.color_palette("viridis", top_n)[::-1],
            edgecolor="white",
        )
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=16, fontweight="bold")
        ax.set_xlabel("Importance")

        # Annotate
        for bar, val in zip(bars, top_features["importance"][::-1]):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_importance.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_forecast(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        last_n_days: int = 90,
    ) -> str:
        """
        Plot the sales forecast with confidence intervals alongside recent history.
        """
        fig, ax = plt.subplots(figsize=(16, 7))

        # Recent historical data
        hist_dates = pd.to_datetime(historical_df["date"])
        hist_sales = historical_df["sales"].values

        if last_n_days < len(hist_dates):
            hist_dates = hist_dates.iloc[-last_n_days:]
            hist_sales = hist_sales[-last_n_days:]

        # Forecast data
        fc_dates = pd.to_datetime(forecast_df["date"])
        fc_sales = forecast_df["predicted_sales"].values
        fc_lower = forecast_df["lower_bound"].values
        fc_upper = forecast_df["upper_bound"].values

        # Plot historical
        ax.plot(hist_dates, hist_sales, color="steelblue", linewidth=1.2,
                label="Historical Sales", alpha=0.8)

        # Plot 30-day rolling average of historical
        rolling = pd.Series(hist_sales).rolling(7, min_periods=1).mean()
        ax.plot(hist_dates, rolling, color="navy", linewidth=2,
                label="7-Day Moving Avg", alpha=0.6)

        # Plot forecast
        ax.plot(fc_dates, fc_sales, color="red", linewidth=2.5,
                label="Forecast", marker="o", markersize=3)

        # Confidence interval
        ax.fill_between(
            fc_dates, fc_lower, fc_upper,
            alpha=0.2, color="red", label="95% Confidence Interval"
        )

        # Vertical line at forecast start
        ax.axvline(
            fc_dates.iloc[0], color="gray", linestyle="--", alpha=0.7,
            label="Forecast Start"
        )

        ax.set_title("Sales Forecast", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend(loc="upper left", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=45)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "forecast_plot.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """
        Plot correlation heatmap of numerical features.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Limit to key columns for readability
        key_cols = [
            c for c in numeric_cols
            if not c.startswith("sales_rolling_") and not c.startswith("sales_ema_")
            and not c.endswith("_sin") and not c.endswith("_cos")
        ][:15]

        if len(key_cols) < 3:
            key_cols = numeric_cols[:15]

        corr = df[key_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "correlation_heatmap.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    def plot_residual_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> str:
        """
        Residual analysis plots for the best model.
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Residual Analysis — {model_name}", fontsize=16, fontweight="bold")

        # Residuals vs Predicted
        ax = axes[0]
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
        ax.axhline(0, color="red", linewidth=1)
        ax.set_title("Residuals vs Predicted")
        ax.set_xlabel("Predicted Sales")
        ax.set_ylabel("Residual")

        # Residual distribution
        ax = axes[1]
        ax.hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linewidth=1)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")

        # Q-Q plot
        ax = axes[2]
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, alpha=0.5, s=10, color="steelblue")
        ax.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=2)
        ax.set_title("Q-Q Plot")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "residual_analysis.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path


if __name__ == "__main__":
    from data_generator import generate_sales_data

    df = generate_sales_data(num_days=365)
    viz = SalesVisualizer(output_dir="results")
    viz.plot_sales_overview(df)
    viz.plot_sales_decomposition(df)
    print("Visualization test complete.")
