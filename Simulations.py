# -*- coding: utf-8 -*-
"""
temperature_extreme_value_analysis.py

Extreme value analysis of temperature time series data.

This script:
1. Loads historical temperature data from an RDS file.
2. Identifies annual maxima.
3. Fits and visualizes:
   - Gumbel distribution
   - Generalized Extreme Value (GEV) distribution
   - Generalized Pareto (GPD) distribution

Author: Anas Mourahib
Date: 2025-09-28
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import genextreme as gev
from scipy.stats import genpareto as gpd
import pyreadr


# -------------------------------------------------------------------------
# Load and process temperature data from RDS file
# -------------------------------------------------------------------------

def load_rds_data(path: str) -> pd.DataFrame:
    """Load temperature data from an RDS file."""
    result = pyreadr.read_r(path)
    return list(result.values())[0]


def extract_temperature_series(df: pd.DataFrame) -> np.ndarray:
    """Extract temperature series from DataFrame loaded via pyreadr."""
    return np.array([t[0] for t in df.values])


# -------------------------------------------------------------------------
# Extreme value analysis
# -------------------------------------------------------------------------

def find_annual_maxima(temp: np.ndarray, block_size: int = 92) -> list[int]:
    """Identify indices of annual maxima in a time series."""
    n_blocks = len(temp) // block_size
    max_indices = []
    for b in range(n_blocks):
        idx_range = np.arange(b * block_size, (b + 1) * block_size)
        val_block = temp[idx_range]
        idx = np.argmax(val_block)
        max_indices.append(b * block_size + idx)
    return max_indices


def plot_annual_maxima(temp: np.ndarray, max_indices: list[int], years: np.ndarray):
    """Plot time series with annual maxima highlighted."""
    plt.figure()
    plt.plot(temp, color="blue", label="Temperature")
    plt.scatter(max_indices, temp[max_indices], color="red", label="Annual maxima")
    plt.title("Annual Maxima in Temperature Series")
    plt.xlabel("Years")
    plt.ylabel("Temperature")
    plt.xticks(
        np.linspace(0, len(temp) - 1, len(years)),
        labels=years,
        rotation=45
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_histogram(data: np.ndarray, title: str = "Histogram of Annual Maxima"):
    """Plot a histogram of annual maximum temperatures."""
    plt.figure()
    plt.hist(data, color="lightgray", edgecolor="black")
    plt.title(title)
    plt.xlabel("Annual maxima")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Gumbel Distribution Fitting
# -------------------------------------------------------------------------

def gumbel_quantile(p: float) -> float:
    """Quantile function of the standard Gumbel distribution."""
    return -math.log(-math.log(p))


def fit_gumbel(empirical: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a Gumbel distribution via linear regression (QQ plot method)."""
    n = len(empirical)
    probs = np.arange(1, n + 1) / (n + 1)
    gumbel_q = np.array([gumbel_quantile(p) for p in probs])
    model = LinearRegression().fit(gumbel_q.reshape(-1, 1), empirical)
    scale, loc = model.coef_[0], model.intercept_
    return gumbel_q, scale, loc


def plot_qq(empirical: np.ndarray, theoretical: np.ndarray, title: str):
    """Generic Q-Q plot."""
    plt.figure()
    plt.scatter(theoretical, empirical, color="navy")
    plt.plot(
        [min(theoretical), max(theoretical)],
        [min(theoretical), max(theoretical)],
        color="red", linestyle="--"
    )
    plt.title(title)
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Empirical quantiles")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Generalized Extreme Value (GEV) Distribution
# -------------------------------------------------------------------------

def fit_gev(rvs: np.ndarray) -> tuple[float, float, float]:
    """Fit a GEV distribution and return parameters."""
    shape, loc, scale = gev.fit(rvs)
    return -shape, loc, scale  # Negate for consistent convention



# -------------------------------------------------------------------------
# Generalized Pareto (GPD) Distribution
# -------------------------------------------------------------------------


def fit_gpd(rvs: np.ndarray) -> tuple[float, float, float]:
    """Fit a GEV distribution and return parameters."""
    shape, loc, scale = gpd.fit(rvs)
    return shape, loc, scale  # Negate for consistent convention

# -------------------------------------------------------------------------
# Generalized Pareto Distribution (GPD)
# -------------------------------------------------------------------------

def find_exceedances(temp: np.ndarray, quantile: float = 0.95) -> tuple[float, np.ndarray]:
    """Return exceedances above a given quantile threshold."""
    threshold = np.quantile(temp, q=quantile)
    exceedances = temp[temp > threshold]
    return threshold, exceedances


def plot_exceedances(temp: np.ndarray, threshold: float, exceedances: np.ndarray):
    """Plot time series with exceedances highlighted."""
    plt.figure()
    plt.plot(temp, color="steelblue")
    plt.scatter(np.where(temp > threshold), exceedances, color="red", label="Exceedances")
    plt.axhline(y=threshold, color="black", linestyle="--", label="Threshold")
    plt.title("Exceedances above 0.95 Quantile")
    plt.xlabel("Time index")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------------

def main():
    """Run the full extreme value analysis pipeline."""
    # Load and extract temperature data
    rds_path = "C:/Users/20254817/Desktop/Githib/Simulations/st1.rds"
    rds_df = load_rds_data(rds_path)
    temp_series = extract_temperature_series(rds_df)

    # Compute and visualize annual maxima
    years = np.arange(1850, 2020, step=20)
    max_indices = find_annual_maxima(temp_series)
    plot_annual_maxima(temp_series, max_indices, years)
    plot_histogram(temp_series[max_indices])

    # Fit and visualize Gumbel distribution
    gumbel_q, scale, loc = fit_gumbel(np.sort(temp_series[max_indices]))
    plot_qq(np.sort(temp_series[max_indices]), gumbel_q,
            "Q-Q Plot: Empirical vs. Standard Gumbel")

    # Fit and display GEV distribution parameters
    shape, loc, scale = fit_gev(temp_series[max_indices])
    print(f"GEV parameters: shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")

    # Identify and plot exceedances (GPD step)
    threshold, exceedances = find_exceedances(temp_series)
    plot_exceedances(temp_series, threshold, exceedances)
    stand_exceedances = exceedances - threshold
    
    shape, loc, scale = fit_gpd(stand_exceedances)
    print(f"GPD parameters: shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")

if __name__ == "__main__":
    main()
