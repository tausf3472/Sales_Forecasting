"""
Model Trainer for Sales Forecasting
Trains multiple ML models, evaluates performance, and selects the best model.
"""

import numpy as np
import pandas as pd
import warnings
import joblib
from typing import Dict, Tuple, Optional

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")


class SalesModelTrainer:
    """
    Trains and evaluates multiple regression models for sales forecasting.

    Supported models:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost (if installed)
    - LSTM (optional, requires TensorFlow)
    """

    def __init__(self, use_lstm: bool = True):
        """
        Parameters
        ----------
        use_lstm : bool
            Whether to include LSTM model in training.
        """
        self.use_lstm = use_lstm
        self.models: Dict[str, object] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[object] = None
        self.feature_importances: Optional[pd.DataFrame] = None

    def _build_models(self) -> Dict[str, object]:
        """Create dictionary of models to train."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
        }

        # Try to add XGBoost
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            print("[INFO] XGBoost not installed. Skipping XGBoost model.")

        return models

    def _evaluate_model(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Returns
        -------
        Dict with MAE, RMSE, MAPE, R2 scores.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE — handle zero values
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = float("inf")

        return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "R2": round(r2, 4)}

    def train_traditional_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all traditional ML models and evaluate on test set.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_test, y_test : Test data.

        Returns
        -------
        Dict mapping model name -> metrics dict.
        """
        models = self._build_models()
        predictions = {}

        print("\n" + "=" * 60)
        print("  TRAINING MACHINE LEARNING MODELS")
        print("=" * 60)

        for name, model in models.items():
            print(f"\n  Training {name}...", end=" ")
            try:
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)

                # Ensure non-negative predictions
                y_pred = np.maximum(y_pred, 0)

                metrics = self._evaluate_model(y_test.values, y_pred)
                self.models[name] = model
                self.results[name] = metrics
                predictions[name] = y_pred

                print(f"Done  |  MAE: {metrics['MAE']:.2f}  RMSE: {metrics['RMSE']:.2f}  "
                      f"MAPE: {metrics['MAPE']:.2f}%  R2: {metrics['R2']:.4f}")

            except Exception as e:
                print(f"Failed: {e}")

        # Extract feature importances from tree-based models
        self._extract_feature_importances(X_train.columns.tolist())

        self.predictions = predictions
        return self.results

    def train_lstm_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        preprocessor=None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Optional[Dict[str, float]]:
        """
        Train an LSTM neural network for time series forecasting.

        Parameters
        ----------
        X_train, y_train : Training sequences (3D arrays).
        X_test, y_test : Test sequences (3D arrays).
        preprocessor : SalesDataPreprocessor instance (for inverse scaling).
        epochs : Number of training epochs.
        batch_size : Training batch size.

        Returns
        -------
        Dict of metrics, or None if TensorFlow is not available.
        """
        if not self.use_lstm:
            print("\n  [INFO] LSTM training skipped (--no-lstm flag).")
            return None

        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping

            tf.get_logger().setLevel("ERROR")
        except ImportError:
            print("\n  [INFO] TensorFlow not installed. Skipping LSTM model.")
            return None

        print(f"\n  Training LSTM Neural Network...", end=" ")

        n_features = X_train.shape[2]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0,
        )

        # Predict and inverse transform
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()

        if preprocessor is not None:
            y_pred = preprocessor.inverse_transform_target(y_pred_scaled)
            y_true = preprocessor.inverse_transform_target(y_test)
        else:
            y_pred = y_pred_scaled
            y_true = y_test

        y_pred = np.maximum(y_pred, 0)

        metrics = self._evaluate_model(y_true, y_pred)
        self.models["LSTM"] = model
        self.results["LSTM"] = metrics
        self.predictions["LSTM"] = y_pred
        self.lstm_history = history

        print(f"Done  |  MAE: {metrics['MAE']:.2f}  RMSE: {metrics['RMSE']:.2f}  "
              f"MAPE: {metrics['MAPE']:.2f}%  R2: {metrics['R2']:.4f}")

        return metrics

    def _extract_feature_importances(self, feature_names: list):
        """Extract and store feature importances from tree-based models."""
        importance_data = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importance_data[name] = model.feature_importances_

        if importance_data:
            # Use the best tree-based model's importances
            best_tree = max(
                importance_data.keys(),
                key=lambda k: self.results[k]["R2"],
            )
            importances = importance_data[best_tree]
            self.feature_importances = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

    def select_best_model(self, metric: str = "RMSE") -> Tuple[str, object]:
        """
        Select the best model based on a given metric.

        Parameters
        ----------
        metric : str
            Metric to use for selection ('MAE', 'RMSE', 'MAPE', 'R2').

        Returns
        -------
        Tuple of (model_name, model_object).
        """
        if not self.results:
            raise ValueError("No models have been trained yet.")

        if metric == "R2":
            # Higher is better
            self.best_model_name = max(
                self.results, key=lambda k: self.results[k][metric]
            )
        else:
            # Lower is better
            self.best_model_name = min(
                self.results, key=lambda k: self.results[k][metric]
            )

        self.best_model = self.models[self.best_model_name]

        print(f"\n{'=' * 60}")
        print(f"  BEST MODEL: {self.best_model_name}")
        print(f"  {metric}: {self.results[self.best_model_name][metric]}")
        print(f"{'=' * 60}")

        return self.best_model_name, self.best_model

    def get_results_dataframe(self) -> pd.DataFrame:
        """Return model comparison results as a DataFrame."""
        df = pd.DataFrame(self.results).T
        df.index.name = "Model"
        return df.sort_values("RMSE")

    def save_best_model(self, filepath: str = "best_model.pkl"):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model() first.")

        if self.best_model_name == "LSTM":
            self.best_model.save(filepath.replace(".pkl", ".keras"))
            print(f"  LSTM model saved to {filepath.replace('.pkl', '.keras')}")
        else:
            joblib.dump(self.best_model, filepath)
            print(f"  Model saved to {filepath}")


if __name__ == "__main__":
    from data_generator import generate_sales_data
    from data_preprocessor import SalesDataPreprocessor

    # Generate data
    df = generate_sales_data(num_days=730)

    # Preprocess
    preprocessor = SalesDataPreprocessor(test_size=0.2)
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(df)

    # Train models
    trainer = SalesModelTrainer(use_lstm=False)
    results = trainer.train_traditional_models(X_train, y_train, X_test, y_test)

    # Select best
    best_name, best_model = trainer.select_best_model(metric="RMSE")

    # Show comparison
    print("\n\nModel Comparison:")
    print(trainer.get_results_dataframe().to_string())
