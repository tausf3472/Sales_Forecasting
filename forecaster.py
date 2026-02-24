"""
Sales Forecaster
Generates future sales predictions using the trained best model.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from data_preprocessor import SalesDataPreprocessor


class SalesForecaster:
    """
    Generates future sales forecasts using a trained model.

    Supports iterative multi-step forecasting where each predicted value
    is fed back as input for the next prediction.
    """

    def __init__(
        self,
        model,
        model_name: str,
        preprocessor: SalesDataPreprocessor,
        historical_df: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        model : trained model object
            The best-performing model from training.
        model_name : str
            Name of the model (e.g., 'XGBoost', 'LSTM').
        preprocessor : SalesDataPreprocessor
            Fitted preprocessor with scaler and feature info.
        historical_df : pd.DataFrame
            Original historical sales data (used for lag/rolling features).
        """
        self.model = model
        self.model_name = model_name
        self.preprocessor = preprocessor
        self.historical_df = historical_df.copy()

    def forecast(
        self, days_ahead: int = 30, confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Generate sales forecast for the specified number of days.

        Uses iterative one-step-ahead prediction: each predicted value is
        appended to the history and used to compute features for the next step.

        Parameters
        ----------
        days_ahead : int
            Number of future days to forecast.
        confidence_level : float
            Confidence level for prediction intervals (0-1).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, predicted_sales, lower_bound, upper_bound.
        """
        if self.model_name == "LSTM":
            return self._forecast_lstm(days_ahead, confidence_level)

        return self._forecast_traditional(days_ahead, confidence_level)

    def _forecast_traditional(
        self, days_ahead: int, confidence_level: float
    ) -> pd.DataFrame:
        """Forecast using traditional ML models (iterative approach)."""
        # Work with a copy of historical data that we'll extend
        working_df = self.historical_df.copy()
        last_date = pd.Timestamp(working_df["date"].max())

        predictions = []
        dates = []

        print(f"\n  Generating {days_ahead}-day forecast using {self.model_name}...")

        for step in range(days_ahead):
            next_date = last_date + pd.Timedelta(days=step + 1)

            # Create a row for the next date with placeholder sales
            next_row = self._create_future_row(working_df, next_date)
            working_df = pd.concat([working_df, next_row], ignore_index=True)

            # Engineer features for the entire dataset
            df_feat = self.preprocessor.engineer_features(working_df)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Get the last row's features
            feature_values = df_feat[self.preprocessor.feature_columns].iloc[-1:].values

            # Scale features
            feature_scaled = self.preprocessor.scaler.transform(feature_values)

            # Predict
            pred = self.model.predict(feature_scaled)[0]
            pred = max(pred, 0)  # Ensure non-negative

            # Update the working dataframe with the prediction
            working_df.loc[working_df.index[-1], "sales"] = pred

            predictions.append(pred)
            dates.append(next_date)

        predictions = np.array(predictions)

        # Estimate prediction intervals using historical residuals
        lower, upper = self._compute_confidence_intervals(
            predictions, confidence_level
        )

        forecast_df = pd.DataFrame(
            {
                "date": dates,
                "predicted_sales": np.round(predictions, 2),
                "lower_bound": np.round(lower, 2),
                "upper_bound": np.round(upper, 2),
            }
        )

        print(f"  Forecast complete. Date range: {dates[0].date()} to {dates[-1].date()}")
        print(f"  Average predicted daily sales: {predictions.mean():.2f}")

        return forecast_df

    def _forecast_lstm(
        self, days_ahead: int, confidence_level: float
    ) -> pd.DataFrame:
        """Forecast using LSTM model."""
        try:
            working_df = self.historical_df.copy()
            last_date = pd.Timestamp(working_df["date"].max())

            # Get the last sequence from historical data
            df_feat = self.preprocessor.engineer_features(working_df)
            df_feat = df_feat.dropna().reset_index(drop=True)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

            feature_cols = self.preprocessor.lstm_feature_cols
            seq_len = self.preprocessor.sequence_length

            X_all = df_feat[feature_cols].values
            X_scaled = self.preprocessor.lstm_scaler.transform(X_all)

            # Start with the last sequence
            current_sequence = X_scaled[-seq_len:].copy()

            predictions = []
            dates = []

            print(f"\n  Generating {days_ahead}-day forecast using LSTM...")

            for step in range(days_ahead):
                next_date = last_date + pd.Timedelta(days=step + 1)

                # Reshape for LSTM: (1, seq_len, n_features)
                input_seq = current_sequence.reshape(1, seq_len, -1)

                # Predict (scaled)
                pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]

                # Inverse transform
                pred = self.preprocessor.inverse_transform_target(
                    np.array([pred_scaled])
                )[0]
                pred = max(pred, 0)

                predictions.append(pred)
                dates.append(next_date)

                # Shift sequence and append new prediction features
                # (simplified: repeat last feature row with updated values)
                new_features = current_sequence[-1].copy()
                current_sequence = np.vstack([current_sequence[1:], new_features])

            predictions = np.array(predictions)
            lower, upper = self._compute_confidence_intervals(
                predictions, confidence_level
            )

            forecast_df = pd.DataFrame(
                {
                    "date": dates,
                    "predicted_sales": np.round(predictions, 2),
                    "lower_bound": np.round(lower, 2),
                    "upper_bound": np.round(upper, 2),
                }
            )

            print(f"  Forecast complete. Date range: {dates[0].date()} to {dates[-1].date()}")
            print(f"  Average predicted daily sales: {predictions.mean():.2f}")

            return forecast_df

        except Exception as e:
            print(f"  LSTM forecast failed: {e}. Falling back to traditional forecast.")
            return self._forecast_traditional(days_ahead, confidence_level)

    def _create_future_row(
        self, df: pd.DataFrame, target_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Create a single-row DataFrame for a future date with estimated values.
        """
        row = {
            "date": target_date,
            "sales": 0.0,  # placeholder — will be replaced by prediction
            "promotion": 0,  # assume no promotion by default
            "holiday": self._is_holiday(target_date),
            "temperature": self._estimate_temperature(target_date),
            "day_of_week": target_date.dayofweek,
            "month": target_date.month,
            "is_weekend": int(target_date.dayofweek >= 5),
        }
        return pd.DataFrame([row])

    def _is_holiday(self, date: pd.Timestamp) -> int:
        """Check if a date falls on or near a major holiday."""
        major_holidays = [
            (1, 1), (1, 26), (2, 14), (3, 8), (5, 1), (7, 4),
            (8, 15), (10, 2), (10, 31), (11, 25), (11, 26), (11, 27),
            (12, 24), (12, 25), (12, 26), (12, 31),
        ]
        for hm, hd in major_holidays:
            if date.month == hm and abs(date.day - hd) <= 1:
                return 1
        return 0

    def _estimate_temperature(self, date: pd.Timestamp) -> float:
        """Estimate temperature based on day of year (seasonal pattern)."""
        doy = date.dayofyear
        return 22 + 12 * np.sin(2 * np.pi * (doy - 80) / 365)

    def _compute_confidence_intervals(
        self, predictions: np.ndarray, confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals based on historical forecast error patterns.

        Uses expanding uncertainty: intervals widen as forecast horizon increases.
        """
        from scipy import stats

        n = len(predictions)

        # Base uncertainty from historical sales variability
        historical_std = self.historical_df["sales"].std()

        # Uncertainty grows with forecast horizon (sqrt of time)
        horizon = np.arange(1, n + 1)
        expanding_std = historical_std * 0.15 * np.sqrt(horizon)

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence_level) / 2)

        lower = predictions - z * expanding_std
        upper = predictions + z * expanding_std

        # Ensure non-negative lower bounds
        lower = np.maximum(lower, 0)

        return lower, upper

    def get_forecast_summary(self, forecast_df: pd.DataFrame) -> dict:
        """
        Generate a summary of the forecast.

        Returns
        -------
        dict with summary statistics.
        """
        return {
            "forecast_start": str(forecast_df["date"].min().date()),
            "forecast_end": str(forecast_df["date"].max().date()),
            "days_forecasted": len(forecast_df),
            "avg_predicted_sales": round(forecast_df["predicted_sales"].mean(), 2),
            "min_predicted_sales": round(forecast_df["predicted_sales"].min(), 2),
            "max_predicted_sales": round(forecast_df["predicted_sales"].max(), 2),
            "total_predicted_sales": round(forecast_df["predicted_sales"].sum(), 2),
            "avg_confidence_width": round(
                (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean(), 2
            ),
            "model_used": self.model_name,
        }


if __name__ == "__main__":
    from data_generator import generate_sales_data
    from data_preprocessor import SalesDataPreprocessor
    from model_trainer import SalesModelTrainer

    # Generate and preprocess data
    df = generate_sales_data(num_days=365)
    preprocessor = SalesDataPreprocessor(test_size=0.2)
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(df)

    # Train models
    trainer = SalesModelTrainer(use_lstm=False)
    trainer.train_traditional_models(X_train, y_train, X_test, y_test)
    best_name, best_model = trainer.select_best_model()

    # Forecast
    forecaster = SalesForecaster(best_model, best_name, preprocessor, df)
    forecast_df = forecaster.forecast(days_ahead=14)

    print("\nForecast:")
    print(forecast_df.to_string(index=False))
    print("\nSummary:")
    for k, v in forecaster.get_forecast_summary(forecast_df).items():
        print(f"  {k}: {v}")
