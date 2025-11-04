import pandas as pd
from pathlib import Path
from physicslab.plot_utils import create_figure, save_figure

# Calibration constants (example values, adjust as needed)
N = 50  # Number of turns
L = 0.06  # Length of coil (m)
R_1 = 2500  # Resistance for H (Ohm)
C_2 = 6e-6  # Calibration constant for B (F)
R_2 = 30000  # Resistance for B (Ohm)
n = 150  # Number of coils for B
S = 80e-4  # Cross-sectional area (m^2)


def analyze_magnetization_curve_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the magnetization curve data.

    Algorithms:
        B = (C_2 * R_2) / (n * S) * U_B
        H = N / (L * R_1) * U_H
        mu = B / H
    """
    df = raw_df.copy()
    # Fill B
    if "U_B / mV" in df.columns:
        df["B / T"] = (C_2 * R_2) / (n * S) * df["U_B / mV"]
    # Fill H
    if "U_H / mV" in df.columns:
        df["H / (A * m^-1)"] = N / (L * R_1) * df["U_H / mV"]
    # Fill mu (mu = B / H)
    if "B / T" in df.columns and "H / (A * m^-1)" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["mu / (10^-2 H * m^-1)"] = df["B / T"] / df["H / (A * m^-1)"]
    # Round results
    df["B / T"] = df["B / T"].round(4)
    df["H / (A * m^-1)"] = df["H / (A * m^-1)"].round(2)
    df["mu / (10^-2 H * m^-1)"] = df["mu / (10^-2 H * m^-1)"].round(4)
    return df


def analyze_hysteresis_loop_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the hysteresis loop data.

    Algorithms:
        H = N / (L * R_1) * U_H
        B = (C_2 * R_2) / (n * S) * U_B
    """
    df = raw_df.copy()
    # The input CSV is in "Parameter" rows format, so transpose for easier handling
    if "Parameter" in df.columns:
        df_t = df.set_index("Parameter").T
        # Fill H_m (using U_Hs / mV)
        if "U_Hs / mV" in df_t.columns:
            df_t["H_m / (A * m^-1)"] = (
                N / (L * R_1) * pd.to_numeric(df_t["U_Hs / mV"], errors="coerce")
            )
        # Fill H_c (using U_Hc / mV)
        if "U_Hc / mV" in df_t.columns:
            df_t["H_c / (A * m^-1)"] = (
                N / (L * R_1) * pd.to_numeric(df_t["U_Hc / mV"], errors="coerce")
            )
        # Fill B_c (using U_Bs/ mV)
        if "U_Bs/ mV" in df_t.columns:
            df_t["B_c / T"] = (
                (C_2 * R_2)
                / (n * S)
                * pd.to_numeric(df_t["U_Bs/ mV"], errors="coerce")
                / 10
            )
        # Fill B_r (using U_Br / mV)
        if "U_Br / mV" in df_t.columns:
            df_t["B_r / T"] = (
                (C_2 * R_2)
                / (n * S)
                * pd.to_numeric(df_t["U_Br / mV"], errors="coerce")
                / 10
            )
        # Round results
        df_t["H_m / (A * m^-1)"] = df_t["H_m / (A * m^-1)"].round(2)
        df_t["H_c / (A * m^-1)"] = df_t["H_c / (A * m^-1)"].round(2)
        df_t["B_c / T"] = df_t["B_c / T"].round(4)
        df_t["B_r / T"] = df_t["B_r / T"].round(4)
        # Convert back to original format
        df_out = df_t.T.reset_index().rename(columns={"index": "Parameter"})
        return df_out
    else:
        # If already in wide format, fill columns directly
        return df


def plot_magnetization_curve(processed_df: pd.DataFrame, output_dir: str | Path):
    """
    Plot magnetization curve (hysteresis loop) for both samples from processed hysteresis loop data.
    Uses create_figure and save_figure utilities.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = processed_df.set_index("Parameter")
    sample_names = [
        col
        for col in df.columns
        if "Value" in col or "Sample" in col or "1" in col or "2" in col
    ]
    if not sample_names:
        sample_names = df.columns.tolist()

    fig, ax = create_figure(
        title="磁滞回线 (Hysteresis Loops)",
        xlabel="H / (A·m⁻¹)",
        ylabel="B / T",
    )
    colors = ["b", "g"]
    for idx, sample in enumerate(sample_names):
        try:
            H_m = float(df.at["H_m / (A * m^-1)", sample])
            H_c = float(df.at["H_c / (A * m^-1)", sample])
            B_c = float(df.at["B_c / T", sample])
            B_r = float(df.at["B_r / T", sample])
        except Exception:
            continue

        H = [H_m, H_c, -H_c, -H_m, H_m]
        B = [B_c, B_r, -B_r, -B_c, B_c]

        ax.plot(
            H,
            B,
            marker="o",
            linestyle="-",
            color=colors[idx % len(colors)],
            label=f"Sample {idx + 1}",
        )
        ax.scatter(
            [H_m, H_c, -H_c, -H_m],
            [B_c, B_r, -B_r, -B_c],
            color=colors[idx % len(colors)],
            s=40,
        )

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "magnetization_curve.png")


# def plot_hysteresis_loops(processed_df: pd.DataFrame, output_dir: str | Path):
#     """
#     Plot two hysteresis loops (磁滞回线) for each sample using processed hysteresis loop data.
#     Each loop is determined by H_m, H_c, B_c, B_r.
#     """
#     import matplotlib.pyplot as plt

#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Convert to wide format for easier access
#     df = processed_df.set_index("Parameter")
#     sample_names = [
#         col
#         for col in df.columns
#         if "Value" in col or "Sample" in col or "1" in col or "2" in col
#     ]
#     if not sample_names:
#         sample_names = df.columns.tolist()

#     plt.figure(figsize=(7, 5))
#     colors = ["b", "g"]
#     for idx, sample in enumerate(sample_names):
#         # Extract parameters for this sample
#         try:
#             H_m = float(df.at["H_m / (A * m^-1)", sample])
#             H_c = float(df.at["H_c / (A * m^-1)", sample])
#             B_c = float(df.at["B_c / T", sample])
#             B_r = float(df.at["B_r / T", sample])
#         except Exception:
#             continue

#         # Build the hysteresis loop points (approximate parallelogram)
#         # Four key points: (+H_m, +B_c), (+H_c, +B_r), (-H_c, -B_r), (-H_m, -B_c)
#         H = [H_m, H_c, -H_c, -H_m, H_m]
#         B = [B_c, B_r, -B_r, -B_c, B_c]

#         plt.plot(
#             H,
#             B,
#             marker="o",
#             linestyle="-",
#             color=colors[idx % len(colors)],
#             label=f"Sample {idx + 1}",
#         )

#         # Annotate key points
#         plt.scatter(
#             [H_m, H_c, -H_c, -H_m],
#             [B_c, B_r, -B_r, -B_c],
#             color=colors[idx % len(colors)],
#             s=40,
#         )

#     plt.xlabel("H / (A·m⁻¹)")
#     plt.ylabel("B / T")
#     plt.title("磁滞回线 (Hysteresis Loops)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_dir / "hysteresis_loops.png")
#     plt.close()


def plot_B_vs_H_and_u_vs_H(processed_df: pd.DataFrame, output_dir: str | Path):
    """
    用一个图展示 B vs H 和 mu vs H，显示数据点和拟合曲线（μ-H 拟合为五次多项式，优化配色）。
    B-H 数据点和拟合曲线使用蓝色系，μ-H 使用紫色系。
    修正 μ 轴标签的字体显示问题。
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = processed_df.dropna(
        subset=["B / T", "H / (A * m^-1)", "mu / (10^-2 H * m^-1)"]
    )
    H = df["H / (A * m^-1)"].values
    B = df["B / T"].values
    mu = df["mu / (10^-2 H * m^-1)"].values

    idx = np.argsort(H)
    H_sorted = H[idx]
    B_sorted = B[idx]
    mu_sorted = mu[idx]

    fig, ax1 = create_figure(
        title="B-H曲线与μ-H曲线",
        xlabel="H / (A·m⁻¹)",
        ylabel="B / T",
    )
    ax2 = ax1.twinx()
    # 修正 μ 轴标签的字体显示问题，使用 LaTeX 数学公式
    ax2.set_ylabel(
        r"$\mu\,/\, (10^{-2}\,\mathrm{H}\cdot\mathrm{m}^{-1})$", color="purple"
    )

    # B-H拟合（3次多项式）
    if len(H_sorted) >= 4:
        H_fit = np.linspace(H_sorted.min(), H_sorted.max(), 200)
        B_poly_coef = np.polyfit(H_sorted, B_sorted, 3)
        B_poly = np.polyval(B_poly_coef, H_fit)
        ax1.plot(
            H_fit,
            B_poly,
            label="B-H 拟合曲线",
            color="#1565c0",
            linestyle="--",
            linewidth=2,
        )

    # μ-H拟合（5次多项式）
    if len(H_sorted) >= 6:
        mu_poly_coef = np.polyfit(H_sorted, mu_sorted, 5)
        mu_poly = np.polyval(mu_poly_coef, H_fit)
        ax2.plot(
            H_fit,
            mu_poly,
            label="μ-H 五次拟合",
            color="indigo",
            linestyle="--",
            linewidth=2,
        )
    elif len(H_sorted) >= 4:
        mu_poly_coef = np.polyfit(H_sorted, mu_sorted, 3)
        mu_poly = np.polyval(mu_poly_coef, H_fit)
        ax2.plot(
            H_fit,
            mu_poly,
            label="μ-H 三次拟合",
            color="indigo",
            linestyle="--",
            linewidth=2,
        )

    # 显示数据点
    ax1.scatter(
        H_sorted, B_sorted, color="#42a5f5", marker="o", s=30, label="B-H 数据点"
    )
    ax2.scatter(
        H_sorted, mu_sorted, color="purple", marker="o", s=30, label="μ-H 数据点"
    )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)
    fig.tight_layout()
    save_figure(fig, output_dir / "B_and_mu_vs_H.png")
