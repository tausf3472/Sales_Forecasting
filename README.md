# Sales Forecasting Using Machine Learning

A comprehensive sales forecasting system that uses multiple machine learning models to predict future sales based on historical data. The system generates synthetic sales data with realistic patterns (trends, seasonality, promotions), trains multiple ML models, evaluates their performance, and produces forecasts with visualizations.

## Features

- **Synthetic Data Generation**: Creates realistic sales data with trends, weekly/monthly seasonality, holiday effects, and promotional impacts
- **Data Preprocessing**: Feature engineering including lag features, rolling statistics, date-based features, and promotion encoding
- **Multiple ML Models**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor (XGBoost)
  - LSTM Neural Network (deep learning)
- **Model Evaluation**: Compares models using MAE, RMSE, MAPE, and R² metrics
- **Forecasting**: Generates future sales predictions with confidence intervals
- **Visualization**: Interactive plots for data exploration, model comparison, and forecast results

## Project Structure

```
Sales_Forecasting/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data_generator.py         # Synthetic sales data generation
├── data_preprocessor.py      # Feature engineering & data preparation
├── model_trainer.py          # ML model training & evaluation
├── forecaster.py             # Future sales prediction
├── visualizer.py             # Data & results visualization
└── main.py                   # Main application entry point
```

## Installation

1. Navigate to the project directory:
   ```bash
   cd Sales_Forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the complete pipeline:
```bash
python main.py
```

### Command-Line Options

```bash
python main.py --days 730          # Generate 2 years of data (default: 1095 = 3 years)
python main.py --forecast 60       # Forecast 60 days ahead (default: 30)
python main.py --no-lstm           # Skip LSTM model (faster execution)
python main.py --output results    # Save outputs to 'results' directory
```

### Pipeline Steps

1. **Data Generation** – Creates synthetic daily sales data with realistic patterns
2. **Preprocessing** – Engineers features (lags, rolling means, date features, etc.)
3. **Model Training** – Trains and evaluates multiple ML models
4. **Forecasting** – Predicts future sales using the best-performing model
5. **Visualization** – Generates plots saved to the output directory

## Output

The system produces:
- `sales_data.csv` – Generated sales dataset
- `model_comparison.png` – Bar chart comparing model metrics
- `forecast_plot.png` – Future sales forecast with confidence intervals
- `feature_importance.png` – Top features driving predictions
- `actual_vs_predicted.png` – Model fit on test data
- `sales_decomposition.png` – Trend, seasonality, and residual decomposition

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost
- tensorflow/keras (optional, for LSTM)
- matplotlib, seaborn

## How It Works

### Data Generation
The synthetic data simulates a retail store's daily sales with:
- **Base sales** ~1000 units/day
- **Upward trend** over time
- **Weekly seasonality** (higher on weekends)
- **Monthly seasonality** (peaks at month start/end)
- **Holiday effects** (spikes around major holidays)
- **Promotional events** (random promotions boosting sales)
- **Random noise** for realism

### Feature Engineering
Key features extracted include:
- Day of week, month, quarter, year
- Is weekend / is month start / is month end
- Lag features (1, 7, 14, 30 days)
- Rolling mean and standard deviation (7, 14, 30 day windows)
- Exponential moving averages
- Promotion indicators

### Model Training
Models are trained on 80% of data and evaluated on 20% holdout set. Hyperparameter tuning uses cross-validation for tree-based models.
