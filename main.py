"""
Sales Forecasting Using Machine Learning — Main Application
============================================================
Complete pipeline: data generation → preprocessing → model training →
evaluation → forecasting → visualization.

Usage:
    python main.py
    python main.py --days 730 --forecast 60 --no-lstm --output results
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from data_generator import generate_sales_data
from data_preprocessor import SalesDataPreprocessor
from model_trainer import SalesModelTrainer
from forecaster import SalesForecaster
from visualizer import SalesVisualizer

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sales Forecasting Using Machine Learning"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Number of days of historical data to generate (default: 1095 = 3 years)",
    )
    parser.add_argument(
        "--forecast",
        type=int,
        default=30,
        help="Number of days to forecast ahead (default: 30)",
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="Skip LSTM model training (faster execution)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results and plots (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 60)
    print("   SALES FORECASTING USING MACHINE LEARNING")
    print("=" * 60)


def step_generate_data(num_days: int, seed: int, output_dir: str) -> pd.DataFrame:
    """Step 1: Generate synthetic sales data."""
    print("\n" + "-" * 60)
    print("  STEP 1: Generating Synthetic Sales Data")
    print("-" * 60)

    df = generate_sales_data(num_days=num_days, seed=seed)

    # Save to CSV
    csv_path = os.path.join(output_dir, "sales_data.csv")
    df.to_csv(csv_path, index=False)

    print(f"  Generated {len(df)} days of sales data")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Sales range: {df['sales'].min():.2f} - {df['sales'].max():.2f}")
    print(f"  Mean sales: {df['sales'].mean():.2f}")
    print(f"  Promotion days: {df['promotion'].sum():.0f} ({df['promotion'].mean()*100:.1f}%)")
    print(f"  Holiday days: {df['holiday'].sum():.0f}")
    print(f"  Data saved to: {csv_path}")

    return df


def step_preprocess(df: pd.DataFrame) -> tuple:
    """Step 2: Preprocess data and engineer features."""
    print("\n" + "-" * 60)
    print("  STEP 2: Preprocessing & Feature Engineering")
    print("-" * 60)

    preprocessor = SalesDataPreprocessor(test_size=0.2)
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(df)

    print(f"  Total features engineered: {len(preprocessor.feature_columns)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Train period: {pd.Timestamp(preprocessor.dates_train[0]).date()} to "
          f"{pd.Timestamp(preprocessor.dates_train[-1]).date()}")
    print(f"  Test period:  {pd.Timestamp(preprocessor.dates_test[0]).date()} to "
          f"{pd.Timestamp(preprocessor.dates_test[-1]).date()}")

    # Print feature categories
    lag_feats = [f for f in preprocessor.feature_columns if "lag" in f]
    rolling_feats = [f for f in preprocessor.feature_columns if "rolling" in f]
    ema_feats = [f for f in preprocessor.feature_columns if "ema" in f]
    date_feats = [f for f in preprocessor.feature_columns
                  if f not in lag_feats + rolling_feats + ema_feats
                  and f not in ["promotion", "holiday", "temperature", "temp_squared", "promo_weekend"]]

    print(f"\n  Feature breakdown:")
    print(f"    Date features:    {len(date_feats)}")
    print(f"    Lag features:     {len(lag_feats)}")
    print(f"    Rolling features: {len(rolling_feats)}")
    print(f"    EMA features:     {len(ema_feats)}")
    print(f"    Other features:   {len(preprocessor.feature_columns) - len(date_feats) - len(lag_feats) - len(rolling_feats) - len(ema_feats)}")

    return preprocessor, X_train, y_train, X_test, y_test


def step_train_models(
    preprocessor, X_train, y_train, X_test, y_test, df, use_lstm: bool
) -> tuple:
    """Step 3: Train and evaluate ML models."""
    print("\n" + "-" * 60)
    print("  STEP 3: Training & Evaluating Models")
    print("-" * 60)

    trainer = SalesModelTrainer(use_lstm=use_lstm)

    # Train traditional models
    results = trainer.train_traditional_models(X_train, y_train, X_test, y_test)

    # Train LSTM if requested
    if use_lstm:
        try:
            X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm = (
                preprocessor.prepare_lstm_data(df, sequence_length=30)
            )
            trainer.train_lstm_model(
                X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                preprocessor=preprocessor, epochs=50, batch_size=32,
            )
        except Exception as e:
            print(f"\n  [WARNING] LSTM training failed: {e}")

    # Select best model
    best_name, best_model = trainer.select_best_model(metric="RMSE")

    # Print comparison table
    print("\n  Model Comparison Table:")
    print("  " + "-" * 56)
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R2':>8}")
    print("  " + "-" * 56)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["RMSE"]):
        marker = " *" if name == best_name else "  "
        print(f" {marker}{name:<24} {metrics['MAE']:>8.2f} {metrics['RMSE']:>8.2f} "
              f"{metrics['MAPE']:>7.2f}% {metrics['R2']:>8.4f}")
    print("  " + "-" * 56)

    return trainer, best_name, best_model


def step_forecast(
    best_model, best_name, preprocessor, df, days_ahead: int
) -> pd.DataFrame:
    """Step 4: Generate future sales forecast."""
    print("\n" + "-" * 60)
    print("  STEP 4: Generating Sales Forecast")
    print("-" * 60)

    forecaster = SalesForecaster(best_model, best_name, preprocessor, df)
    forecast_df = forecaster.forecast(days_ahead=days_ahead)

    # Print forecast summary
    summary = forecaster.get_forecast_summary(forecast_df)
    print(f"\n  Forecast Summary:")
    print(f"    Period: {summary['forecast_start']} to {summary['forecast_end']}")
    print(f"    Days forecasted: {summary['days_forecasted']}")
    print(f"    Model used: {summary['model_used']}")
    print(f"    Avg daily sales: {summary['avg_predicted_sales']:.2f}")
    print(f"    Min daily sales: {summary['min_predicted_sales']:.2f}")
    print(f"    Max daily sales: {summary['max_predicted_sales']:.2f}")
    print(f"    Total sales: {summary['total_predicted_sales']:.2f}")
    print(f"    Avg confidence width: +/-{summary['avg_confidence_width']/2:.2f}")

    return forecast_df


def step_visualize(
    df, trainer, preprocessor, forecast_df, output_dir: str
):
    """Step 5: Generate all visualizations."""
    print("\n" + "-" * 60)
    print("  STEP 5: Generating Visualizations")
    print("-" * 60)

    viz = SalesVisualizer(output_dir=output_dir)

    # 1. Sales overview
    viz.plot_sales_overview(df)

    # 2. Sales decomposition
    viz.plot_sales_decomposition(df)

    # 3. Correlation heatmap
    df_feat = preprocessor.engineer_features(df)
    viz.plot_correlation_heatmap(df_feat)

    # 4. Model comparison
    viz.plot_model_comparison(trainer.results)

    # 5. Actual vs Predicted
    y_test = preprocessor.scaler.inverse_transform(
        np.zeros((1, len(preprocessor.feature_columns)))
    )  # dummy — we use raw y_test
    # Get raw y_test from the split
    df_feat_full = preprocessor.engineer_features(df)
    df_feat_full = df_feat_full.dropna().reset_index(drop=True)
    df_feat_full = df_feat_full.replace([np.inf, -np.inf], np.nan).fillna(0)
    y_all = df_feat_full["sales"].values
    split_idx = preprocessor.split_idx
    y_test_raw = y_all[split_idx:]
    dates_test = preprocessor.dates_test

    # Collect predictions — LSTM may have fewer samples due to sequence windowing;
    # the visualizer handles per-model alignment automatically.
    predictions_aligned = {}
    for name, preds in trainer.predictions.items():
        if len(preds) <= len(y_test_raw):
            predictions_aligned[name] = preds

    if predictions_aligned:
        viz.plot_actual_vs_predicted(y_test_raw, predictions_aligned, dates_test)

    # 6. Feature importance
    if trainer.feature_importances is not None:
        viz.plot_feature_importance(trainer.feature_importances, top_n=20)

    # 7. Residual analysis for best model
    if trainer.best_model_name in trainer.predictions:
        best_preds = trainer.predictions[trainer.best_model_name]
        if len(best_preds) == len(y_test_raw):
            viz.plot_residual_analysis(y_test_raw, best_preds, trainer.best_model_name)

    # 8. Forecast plot
    viz.plot_forecast(df, forecast_df, last_n_days=90)

    print(f"\n  All visualizations saved to: {output_dir}/")


def main():
    """Run the complete sales forecasting pipeline."""
    args = parse_args()
    start_time = time.time()

    print_banner()
    print(f"\n  Configuration:")
    print(f"    Historical data: {args.days} days")
    print(f"    Forecast horizon: {args.forecast} days")
    print(f"    LSTM enabled: {not args.no_lstm}")
    print(f"    Output directory: {args.output}")
    print(f"    Random seed: {args.seed}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # --- Pipeline ---
    # Step 1: Generate data
    df = step_generate_data(args.days, args.seed, args.output)

    # Step 2: Preprocess
    preprocessor, X_train, y_train, X_test, y_test = step_preprocess(df)

    # Step 3: Train models
    trainer, best_name, best_model = step_train_models(
        preprocessor, X_train, y_train, X_test, y_test, df,
        use_lstm=not args.no_lstm,
    )

    # Step 4: Forecast
    forecast_df = step_forecast(best_model, best_name, preprocessor, df, args.forecast)

    # Save forecast to CSV
    forecast_csv = os.path.join(args.output, "forecast.csv")
    forecast_df.to_csv(forecast_csv, index=False)
    print(f"\n  Forecast saved to: {forecast_csv}")

    # Step 5: Visualize
    step_visualize(df, trainer, preprocessor, forecast_df, args.output)

    # Save model comparison
    results_csv = os.path.join(args.output, "model_results.csv")
    trainer.get_results_dataframe().to_csv(results_csv)
    print(f"  Model results saved to: {results_csv}")

    # Save best model
    model_path = os.path.join(args.output, "best_model.pkl")
    trainer.save_best_model(model_path)

    # --- Summary ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total execution time: {elapsed:.1f} seconds")
    print(f"  Best model: {best_name}")
    print(f"  Best RMSE: {trainer.results[best_name]['RMSE']:.2f}")
    print(f"  Best R2: {trainer.results[best_name]['R2']:.4f}")
    print(f"  Forecast: {args.forecast} days ahead")
    print(f"  All outputs saved to: {os.path.abspath(args.output)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
