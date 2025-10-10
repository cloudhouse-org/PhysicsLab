# src/scripts/process_sucrose_polarimetry.py

from pathlib import Path
from physicslab.data_utils import load_csv
from physicslab.plot_utils import setup_plot_style
import numpy as np
from physicslab.sucrose_polarimetry.processing import (
    analyze_average_angle_deg,
    fit_slope_from_processed,
    plot_average_angle_vs_concentration,
    calculate_alpha,
    calculate_c_x,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROTATION_ANGLES_CSV = PROJECT_ROOT / "data/raw/sucrose_polarimetry/rotation_angles.csv"
PROCESSED_CSV = PROJECT_ROOT / "data/processed/sucrose_polarimetry/processed_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/sucrose_polarimetry/"


def main() -> None:
    print("Starting the Sucrose Polarimetry data processing workflow...")

    setup_plot_style()

    print(f"\n--> Step 1: Loading raw data from '{ROTATION_ANGLES_CSV.name}'...")
    raw_df = load_csv(ROTATION_ANGLES_CSV)
    if raw_df.empty:
        print(f"Error: No valid data found in '{ROTATION_ANGLES_CSV.name}'.")
        return

    print("\n--> Step 2: Analyzing data...")
    processed_df = analyze_average_angle_deg(raw_df)

    print(f"\n--> Step 3: Saving processed data to '{PROCESSED_CSV.name}'...")
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Processed data saved to '{PROCESSED_CSV}'.")

    print("\n--> Step 4: Calculating slope (B) from numeric data only...")
    m, b = fit_slope_from_processed(processed_df)
    B = m
    if np.isnan(B):
        print("Warning: not enough numeric data to compute slope B.")
    print(f"Computed slope B = {B:.4f}")

    print("\n--> Step 5: Calculating specific rotation and concentration (c_x)...")
    alpha = calculate_alpha(B, 0.2)
    c_x = calculate_c_x(6.59, 0.67, B)
    print(f"Specific rotation (α): {alpha:.4f} °m^2/kg")
    print(f"Concentration (c_x): {c_x:.4f} kg/m^3")

    print(
        "\n--> Step 6: Plotting and saving figures (annotate unknown at computed c_x)..."
    )
    # now call plot with unknown_x_base set to c_x so unknowns are plotted at the computed concentration
    plot_average_angle_vs_concentration(processed_df, OUTPUT_DIR, unknown_x_base=None)

    print("\nSucrose Polarimetry data processing workflow completed successfully!")


if __name__ == "__main__":
    main()
