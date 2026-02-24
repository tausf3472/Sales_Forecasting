"""
Synthetic Sales Data Generator
Generates realistic daily sales data with trends, seasonality, promotions, and noise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sales_data(
    start_date: str = "2022-01-01",
    num_days: int = 1095,
    base_sales: float = 1000.0,
    trend_slope: float = 0.15,
    noise_level: float = 50.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily sales data with realistic patterns.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    num_days : int
        Number of days of data to generate.
    base_sales : float
        Average daily sales baseline.
    trend_slope : float
        Daily increase in sales (linear trend).
    noise_level : float
        Standard deviation of random noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, sales, promotion, holiday, temperature,
        day_of_week, month, is_weekend.
    """
    np.random.seed(seed)

    start = pd.Timestamp(start_date)
    dates = pd.date_range(start=start, periods=num_days, freq="D")

    # --- Component signals ---
    t = np.arange(num_days, dtype=float)

    # Linear trend
    trend = trend_slope * t

    # Weekly seasonality (higher sales on Fri-Sat-Sun)
    day_of_week = np.array([d.dayofweek for d in dates])
    weekly_pattern = np.array([0, -20, -30, -10, 30, 80, 60])  # Mon-Sun
    weekly_seasonality = np.array([weekly_pattern[dow] for dow in day_of_week])

    # Monthly seasonality (peaks at month start and end — payday effect)
    day_of_month = np.array([d.day for d in dates])
    days_in_month = np.array([d.days_in_month for d in dates])
    monthly_seasonality = (
        40 * np.cos(2 * np.pi * day_of_month / days_in_month)
        + 20 * np.cos(4 * np.pi * day_of_month / days_in_month)
    )

    # Yearly seasonality (holiday shopping season Nov-Dec, summer dip)
    day_of_year = np.array([d.dayofyear for d in dates])
    yearly_seasonality = (
        80 * np.cos(2 * np.pi * (day_of_year - 340) / 365)
        + 30 * np.sin(4 * np.pi * day_of_year / 365)
    )

    # --- Promotions ---
    promotions = _generate_promotions(dates, seed)
    promotion_effect = promotions * np.random.uniform(100, 300, size=num_days)

    # --- Holidays ---
    holidays = _generate_holidays(dates)
    holiday_effect = holidays * np.random.uniform(150, 400, size=num_days)

    # --- Temperature (affects foot traffic) ---
    temperature = _generate_temperature(dates, seed)
    temp_effect = -0.5 * (temperature - 22) ** 2 / 50  # optimal around 22°C

    # --- Random noise ---
    noise = np.random.normal(0, noise_level, size=num_days)

    # --- Combine all components ---
    sales = (
        base_sales
        + trend
        + weekly_seasonality
        + monthly_seasonality
        + yearly_seasonality
        + promotion_effect
        + holiday_effect
        + temp_effect
        + noise
    )

    # Ensure non-negative sales
    sales = np.maximum(sales, 0)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "sales": np.round(sales, 2),
            "promotion": promotions.astype(int),
            "holiday": holidays.astype(int),
            "temperature": np.round(temperature, 1),
            "day_of_week": day_of_week,
            "month": np.array([d.month for d in dates]),
            "is_weekend": (day_of_week >= 5).astype(int),
        }
    )

    return df


def _generate_promotions(dates: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """
    Generate random promotional events.
    Promotions last 3-7 days and occur roughly every 2-4 weeks.
    """
    rng = np.random.RandomState(seed + 1)
    n = len(dates)
    promotions = np.zeros(n, dtype=float)

    i = rng.randint(10, 25)
    while i < n:
        duration = rng.randint(3, 8)
        end = min(i + duration, n)
        promotions[i:end] = 1.0
        gap = rng.randint(14, 30)
        i = end + gap

    return promotions


def _generate_holidays(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Mark major holidays and surrounding days.
    """
    holidays = np.zeros(len(dates), dtype=float)

    # Define major holidays (month, day) with a ±1 day window
    major_holidays = [
        (1, 1),    # New Year
        (1, 26),   # Republic Day (India)
        (2, 14),   # Valentine's Day
        (3, 8),    # International Women's Day
        (5, 1),    # May Day
        (7, 4),    # Independence Day (US)
        (8, 15),   # Independence Day (India)
        (10, 2),   # Gandhi Jayanti
        (10, 31),  # Halloween
        (11, 25),  # Black Friday (approx)
        (11, 26),  # Black Friday (approx)
        (11, 27),  # Cyber Monday (approx)
        (12, 24),  # Christmas Eve
        (12, 25),  # Christmas
        (12, 26),  # Boxing Day
        (12, 31),  # New Year's Eve
    ]

    for idx, d in enumerate(dates):
        for hm, hd in major_holidays:
            if d.month == hm and abs(d.day - hd) <= 1:
                holidays[idx] = 1.0
                break

    return holidays


def _generate_temperature(dates: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """
    Generate synthetic daily temperature with seasonal variation.
    Simulates a Northern Hemisphere temperate climate.
    """
    rng = np.random.RandomState(seed + 2)
    day_of_year = np.array([d.dayofyear for d in dates])

    # Seasonal temperature: peaks in summer (~July), troughs in winter (~Jan)
    seasonal_temp = 22 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Add daily random variation
    daily_noise = rng.normal(0, 3, size=len(dates))

    return seasonal_temp + daily_noise


if __name__ == "__main__":
    df = generate_sales_data()
    print(f"Generated {len(df)} days of sales data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSales statistics:")
    print(df["sales"].describe())
    print(f"\nPromotion days: {df['promotion'].sum():.0f}")
    print(f"Holiday days: {df['holiday'].sum():.0f}")
    print(f"\nFirst 5 rows:")
    print(df.head())
