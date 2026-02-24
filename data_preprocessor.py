"""
Data Preprocessor for Sales Forecasting
Handles feature engineering, scaling, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional


class SalesDataPreprocessor:
    """
    Preprocesses raw sales data for machine learning models.

    Performs feature engineering (lag features, rolling statistics, date features),
    handles missing values, scales features, and splits data into train/test sets.
    """

    def __init__(self, target_col: str = "sales", test_size: float = 0.2):
        """
        Parameters
        ----------
        target_col : str
            Name of the target column.
        test_size : float
            Fraction of data to use for testing (from the end, preserving time order).
        """
        self.target_col = target_col
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = []
        self._is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw sales data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw sales data with 'date' and 'sales' columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # --- Date-based features ---
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

        # --- Cyclical encoding for periodic features ---
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # --- Lag features ---
        lag_days = [1, 2, 3, 7, 14, 21, 28, 30]
        for lag in lag_days:
            df[f"sales_lag_{lag}"] = df[self.target_col].shift(lag)

        # --- Rolling window statistics ---
        windows = [7, 14, 30]
        for window in windows:
            df[f"sales_rolling_mean_{window}"] = (
                df[self.target_col].shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"sales_rolling_std_{window}"] = (
                df[self.target_col].shift(1).rolling(window=window, min_periods=1).std()
            )
            df[f"sales_rolling_min_{window}"] = (
                df[self.target_col].shift(1).rolling(window=window, min_periods=1).min()
            )
            df[f"sales_rolling_max_{window}"] = (
                df[self.target_col].shift(1).rolling(window=window, min_periods=1).max()
            )

        # --- Exponential moving averages ---
        for span in [7, 14, 30]:
            df[f"sales_ema_{span}"] = (
                df[self.target_col].shift(1).ewm(span=span, adjust=False).mean()
            )

        # --- Difference features ---
        df["sales_diff_1"] = df[self.target_col].diff(1)
        df["sales_diff_7"] = df[self.target_col].diff(7)

        # --- Percentage change ---
        df["sales_pct_change_1"] = df[self.target_col].pct_change(1)
        df["sales_pct_change_7"] = df[self.target_col].pct_change(7)

        # --- Interaction features ---
        if "promotion" in df.columns:
            df["promo_weekend"] = df["promotion"] * df["is_weekend"]

        if "temperature" in df.columns:
            df["temp_squared"] = df["temperature"] ** 2

        return df

    def prepare_data(
        self, df: pd.DataFrame, drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Full preprocessing pipeline: feature engineering, cleaning, splitting, scaling.

        Parameters
        ----------
        df : pd.DataFrame
            Raw sales data.
        drop_na : bool
            Whether to drop rows with NaN values (from lag/rolling features).

        Returns
        -------
        Tuple of (X_train, y_train, X_test, y_test)
        """
        # Engineer features
        df_feat = self.engineer_features(df)

        # Drop NaN rows created by lag/rolling features
        if drop_na:
            df_feat = df_feat.dropna().reset_index(drop=True)

        # Define feature columns (exclude date and target)
        exclude_cols = {"date", self.target_col, "sales_diff_1", "sales_diff_7",
                        "sales_pct_change_1", "sales_pct_change_7"}
        self.feature_columns = [
            col for col in df_feat.columns if col not in exclude_cols
        ]

        # Replace any remaining infinities
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        df_feat = df_feat.fillna(0)

        X = df_feat[self.feature_columns].values
        y = df_feat[self.target_col].values
        dates = df_feat["date"].values

        # Time-based split (no shuffling — preserve temporal order)
        split_idx = int(len(X) * (1 - self.test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Store dates for later use
        self.dates_train = dates_train
        self.dates_test = dates_test
        self.split_idx = split_idx
        self._is_fitted = True

        # Convert back to DataFrames for convenience
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        y_train_series = pd.Series(y_train, name=self.target_col)
        y_test_series = pd.Series(y_test, name=self.target_col)

        return X_train_df, y_train_series, X_test_df, y_test_series

    def prepare_lstm_data(
        self, df: pd.DataFrame, sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data specifically for LSTM model (3D input: samples, timesteps, features).

        Parameters
        ----------
        df : pd.DataFrame
            Raw sales data.
        sequence_length : int
            Number of past time steps to use as input sequence.

        Returns
        -------
        Tuple of (X_train, y_train, X_test, y_test) with LSTM-compatible shapes.
        """
        df_feat = self.engineer_features(df)
        df_feat = df_feat.dropna().reset_index(drop=True)

        exclude_cols = {"date", self.target_col, "sales_diff_1", "sales_diff_7",
                        "sales_pct_change_1", "sales_pct_change_7"}
        feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

        # Replace infinities and NaN
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

        X_all = df_feat[feature_cols].values
        y_all = df_feat[self.target_col].values

        # Scale
        lstm_scaler = StandardScaler()
        X_scaled = lstm_scaler.fit_transform(X_all)

        # Scale target for LSTM
        y_scaled = self.target_scaler.fit_transform(y_all.reshape(-1, 1)).flatten()

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i - sequence_length : i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Time-based split
        split_idx = int(len(X_seq) * (1 - self.test_size))
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]

        self.lstm_scaler = lstm_scaler
        self.lstm_feature_cols = feature_cols
        self.sequence_length = sequence_length

        return X_train, y_train, X_test, y_test

    def get_feature_names(self) -> list:
        """Return the list of feature column names."""
        return self.feature_columns

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform LSTM-scaled target values."""
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()


if __name__ == "__main__":
    from data_generator import generate_sales_data

    df = generate_sales_data(num_days=365)
    preprocessor = SalesDataPreprocessor(test_size=0.2)

    X_train, y_train, X_test, y_test = preprocessor.prepare_data(df)

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"\nFeature columns ({len(preprocessor.feature_columns)}):")
    for col in preprocessor.feature_columns:
        print(f"  - {col}")
