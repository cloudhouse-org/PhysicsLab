from pathlib import Path
import pandas as pd
from physicslab.magnetization_curve.processing import (
    analyze_magnetization_curve_data,
    analyze_hysteresis_loop_data,
    plot_B_vs_H_and_u_vs_H,
)
from physicslab.plot_utils import setup_plot_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw/magnitization_curve"
PROCESSED_DIR = PROJECT_ROOT / "data/processed/magnitization_curve"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_INIT_CSV = RAW_DIR / "initial_magnetization_curve_data.csv"
RAW_HYST_CSV = RAW_DIR / "hysteresis_loop_data.csv"
PROC_INIT_CSV = PROCESSED_DIR / "initial_magnetization_curve_data.csv"
PROC_HYST_CSV = PROCESSED_DIR / "hysteresis_loop_data.csv"


def main():
    setup_plot_style()

    print("Processing initial magnetization curve data...")
    df_init = pd.read_csv(RAW_INIT_CSV)
    df_init_proc = analyze_magnetization_curve_data(df_init)
    df_init_proc.to_csv(PROC_INIT_CSV, index=False)
    print(f"Saved processed initial magnetization curve to {PROC_INIT_CSV}")

    print("Processing hysteresis loop data...")
    df_hyst = pd.read_csv(RAW_HYST_CSV)
    df_hyst_proc = analyze_hysteresis_loop_data(df_hyst)
    df_hyst_proc.to_csv(PROC_HYST_CSV, index=False)
    print(f"Saved processed hysteresis loop to {PROC_HYST_CSV}")

    # print("Plotting magnetization curve...")
    # plot_magnetization_curve(df_hyst_proc, PROJECT_ROOT / "output/magnetization_curve/")
    # print("Plotting complete.")

    print("Plotting B vs H and u vs H...")
    plot_B_vs_H_and_u_vs_H(
        df_init_proc, PROJECT_ROOT / "output/magnetization_curve/"
    )
    print("Plotting complete.")


if __name__ == "__main__":
    main()
