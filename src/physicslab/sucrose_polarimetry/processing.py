# src/physicslab/sucrose_polarimetry/processing.py

import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from physicslab.plot_utils import create_figure, save_figure


def analyze_average_angle_deg(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sucrose polarimetry data to compute average angles.

    Parameters:
    raw_df (pd.DataFrame): DataFrame containing raw data with columns 'trial1_deg', 'trial2_deg', 'trial3_deg'.

    Returns:
    pd.DataFrame: DataFrame with computed average angles.
    """
    df = raw_df.copy()

    # For sucrose polarimetry data
    trial_cols = [
        "trial1_left",
        "trial2_left",
        "trial3_left",
        "trial4_left",
        "trial5_left",
        "trial1_right",
        "trial2_right",
        "trial3_right",
        "trial4_right",
        "trial5_right",
    ]
    if all(col in df.columns for col in trial_cols):
        # Ignore the first row where sucrose_concentration is 0
        mask = df["sucrose_concentration_kg_m3"] != 0
        # Fill average_angle_deg as mean of trial1_deg, trial2_deg, trial3_deg for rows where sucrose_concentration != 0
        df.loc[mask, "average_angle_deg"] = df.loc[mask, trial_cols].mean(axis=1)
        # Round to two decimal places
        df["average_angle_deg"] = df["average_angle_deg"].round(2)
        df["angle_deg"] = df["average_angle_deg"].round(2)

    return df


def calculate_alpha(B: float, i: float) -> float:
    return B / (i * 1.0)


def calculate_c_x(phi_x: float, phi_0: float, B: float) -> float:
    return (phi_x - phi_0) / B


def plot_average_angle_vs_concentration(
    processed_df: pd.DataFrame,
    output_dir: str | Path,
    unknown_x_base: float | None = None,
) -> float:
    """
    Plot average angle vs sucrose concentration and save the figure.
    Return the slope of the fitted line (specific rotation).

    MODIFIED LOGIC: When `unknown_x_base` is not specified, the x-coordinate for
    an unknown sample is calculated from its measured angle (phi_x) using the
    fitted linear model. This places the unknown point directly on the fitted line.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = processed_df.copy()
    # preserve raw concentration column for detecting 'unknown'
    if "sucrose_concentration_kg_m3" in df.columns:
        df["sucrose_concentration_raw"] = df["sucrose_concentration_kg_m3"]
    else:
        df["sucrose_concentration_raw"] = df.get("sucrose_concentration", np.nan)

    # detect rows explicitly labeled 'unknown' (case-insensitive)
    unknown_mask = (
        df["sucrose_concentration_raw"].astype(str).str.strip().str.lower() == "unknown"
    )

    # Convert concentration to numeric for fitting (unknowns -> NaN)
    df["sucrose_concentration"] = pd.to_numeric(
        df["sucrose_concentration_raw"], errors="coerce"
    )
    df["average_angle_deg"] = pd.to_numeric(df["average_angle_deg"], errors="coerce")

    # Data used for fitting: numeric concentrations only (unknowns excluded)
    df_fit = df[~unknown_mask].dropna(
        subset=["average_angle_deg", "sucrose_concentration"]
    )
    # Exclude rows where concentration == 0 (blank/control) from fit and plotting
    df_fit = df_fit[df_fit["sucrose_concentration"] != 0]
    x1 = df_fit["sucrose_concentration"]
    y1 = df_fit["average_angle_deg"]

    fig, ax1 = create_figure(
        title="平均旋光角与蔗糖浓度关系",
        xlabel="$c/(kg/m³)$",
        ylabel="$\\phi$",
    )

    # Plot numeric data points
    if not df_fit.empty:
        ax1.scatter(x1, y1, color="blue", label="数据点")

    # Fit a linear model using only numeric rows
    def linear_model(x, m, b):
        return m * x + b

    m = float("nan")  # Default slope if fitting fails
    b = float("nan")  # Default intercept if fitting fails
    if len(x1) >= 2:  # Ensure there are enough points to fit
        try:
            popt, pcov = curve_fit(linear_model, x1, y1)
            m, b = popt
            x_fit = np.linspace(float(x1.min()), float(x1.max()), 100)
            y_fit = linear_model(x_fit, m, b)
            ax1.plot(x_fit, y_fit, color="red", label=f"拟合: y={m:.4f}x+{b:.2f}")
        except (RuntimeError, ValueError):
            # Fitting can fail if data is unsuitable
            print("Warning: Linear fitting failed. Proceeding without a fitted line.")
            m = float("nan")
    else:
        print(
            "Warning: Not enough data points for fitting. Proceeding without a fitted line."
        )

    # Plot unknown points
    unknown_points = df[unknown_mask & df["average_angle_deg"].notna()].copy()
    if not unknown_points.empty:
        added_unknown_label = False

        # Pre-calculate values needed for fallback placement
        if not df_fit.empty:
            x_min, x_max = float(x1.min()), float(x1.max())
            span = x_max - x_min if x_max > x_min else 1.0
            fallback_x = x_max + 0.05 * span
        else:
            span = 1.0  # Default span if no fit data
            fallback_x = 0.0

        for _, row in unknown_points.iterrows():
            y_unknown = float(row["average_angle_deg"])

            # --- MODIFIED LOGIC START ---
            # Determine the x-coordinate for the unknown point
            if unknown_x_base is not None:
                # Case 1: User explicitly provided an x-coordinate
                x_coord_for_plot = float(unknown_x_base)
            else:
                # Case 2: x-coordinate not specified, calculate from the fitted line
                # Check if we have a valid fit (m is not NaN and not zero)
                if not np.isnan(m) and m != 0:
                    # Calculate x from y = mx + b  =>  x = (y - b) / m
                    x_coord_for_plot = (y_unknown - b) / m
                else:
                    # Fallback: No valid fit, use the original default placement
                    x_coord_for_plot = fallback_x
            # --- MODIFIED LOGIC END ---

            # Plot the unknown point
            ax1.scatter(
                x_coord_for_plot,
                y_unknown,
                marker="s",
                color="black",
                label=("未知" if not added_unknown_label else None),
            )
            # Draw dashed lines to indicate its x and y positions
            ax1.axvline(x_coord_for_plot, linestyle="--", color="gray", linewidth=0.8)
            ax1.axhline(y_unknown, linestyle="--", color="gray", linewidth=0.8)

            # Annotate with the calculated/provided c_x value
            text_x_offset = 0.02 * span
            ax1.text(
                x_coord_for_plot + text_x_offset,
                y_unknown,
                f"$c_x$ = {x_coord_for_plot:.2f}",
                fontsize=9,
                va="center",
            )
            added_unknown_label = True

    ax1.legend()
    save_figure(fig, output_dir / "average_angle_vs_concentration.png")

    return float(m)


def fit_slope_from_processed(processed_df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute linear fit slope and intercept (m, b) from processed DataFrame
    excluding rows labeled 'unknown' and concentrations equal to 0.

    Returns (m, b). If not enough data, returns (nan, nan).
    """
    df = processed_df.copy()
    if "sucrose_concentration_kg_m3" in df.columns:
        df["sucrose_concentration_raw"] = df["sucrose_concentration_kg_m3"]
    else:
        df["sucrose_concentration_raw"] = df.get("sucrose_concentration", np.nan)

    unknown_mask = (
        df["sucrose_concentration_raw"].astype(str).str.strip().str.lower() == "unknown"
    )
    df["sucrose_concentration"] = pd.to_numeric(
        df["sucrose_concentration_raw"], errors="coerce"
    )
    df["average_angle_deg"] = pd.to_numeric(df["average_angle_deg"], errors="coerce")

    df_fit = df[~unknown_mask].dropna(
        subset=["average_angle_deg", "sucrose_concentration"]
    )
    df_fit = df_fit[df_fit["sucrose_concentration"] != 0]

    if len(df_fit) < 2:
        return float("nan"), float("nan")

    x = df_fit["sucrose_concentration"].astype(float)
    y = df_fit["average_angle_deg"].astype(float)
    popt, _ = curve_fit(lambda x, m, b: m * x + b, x, y)
    m, b = popt
    return float(m), float(b)
