#!/usr/bin/env python3
"""
Time series visualization and CSV export

Generates plots and simplified CSV output for:
- Rain attenuation
- Energy harvested
- Battery SoC & thermal state
- Safety metrics (PFD, interference, thermal)
- Objective value
- Action space variables
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from params import SimParams
import simulate

SMOOTH_WINDOW = 15

def _smooth_array(values, window_size: int = SMOOTH_WINDOW) -> np.ndarray:
    s = pd.Series(np.asarray(values))
    return s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

def _linestyle_map(policies: List[str]) -> Dict[str, str]:
    # Distinct linestyles without specifying colors.
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 5))]
    return {pol: styles[i % len(styles)] for i, pol in enumerate(policies)}

def _safe_plot(ax, t, series, label, linestyle):
    if series is None:
        return
    arr = np.asarray(series)
    if arr.size == 0:
        return
    y = _smooth_array(arr)
    ax.plot(t, y, linestyle=linestyle, label=label)

def export_simplified_csv(dfs_by_policy: Dict[str, pd.DataFrame], out_dir: Path, P: SimParams):
    """Export simplified CSV with key metrics for each policy"""
    for policy, df in dfs_by_policy.items():
        # Select key columns for simplified output
        columns_to_export = [
            't', 'rain_db', 'E_harv_J', 'S_t', 'T_thermal_C', 'T_next',
            'pfd_max_Wm2', 'I_wpt_dBm', 'objective_value', 'J_t',
            'P_tx_W', 'tau', 'u', 'sinr_db', 'latency_ms'
        ]

        # Filter to only existing columns
        available_cols = [col for col in columns_to_export if col in df.columns]

        # Create simplified dataframe
        df_simplified = df[available_cols].copy()

        # Rename for clarity
        rename_map = {
            't': 'time_slot',
            'rain_db': 'rain_attenuation_dB',
            'E_harv_J': 'energy_harvest_J',
            'S_t': 'battery_soc',
            'T_thermal_C': 'temperature_C',
            'T_next': 'temperature_C',
            'pfd_max_Wm2': 'power_flux_density_Wm2',
            'I_wpt_dBm': 'interference_dBm',
            'objective_value': 'objective',
            'J_t': 'cost',
            'P_tx_W': 'tx_power_W',
            'tau': 'duty_cycle',
            'u': 'wpt_active',
            'sinr_db': 'sinr_dB',
            'latency_ms': 'latency_ms'
        }

        df_simplified.rename(columns=rename_map, inplace=True)

        # Save to CSV
        csv_path = out_dir / f"timeseries_{policy}_simplified.csv"
        df_simplified.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"[saved] {csv_path}")
        print(f"  Columns: {list(df_simplified.columns)}")
        print(f"  Rows: {len(df_simplified)}")

def main():
    P = SimParams()
    out_dir = Path(P.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run episodes for all policies
    dfs_by_policy: Dict[str, pd.DataFrame] = simulate.run_episodes(P)

    policies = [pol for pol in P.policies if pol in dfs_by_policy]
    if not policies:
        policies = list(dfs_by_policy.keys())

    # Export simplified CSV files
    export_simplified_csv(dfs_by_policy, out_dir, P)

    style_map = _linestyle_map(policies)

    # Build visualization figure
    fig, axes = plt.subplots(10, 1, figsize=(15, 24), sharex=True, constrained_layout=True)

    def get_t(df: pd.DataFrame):
        return df["t"].to_numpy() if "t" in df.columns else np.arange(len(df))

    # (1) Rain attenuation
    ax = axes[0]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "rain_db" in df.columns:
            _safe_plot(ax, t, df["rain_db"], f"{pol} rain", style_map[pol])
    ax.set_ylabel("Rain [dB]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Rain Attenuation")

    # (2) Energy Harvested
    ax = axes[1]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "E_harv_J" in df.columns:
            _safe_plot(ax, t, df["E_harv_J"], f"{pol} E_harv", style_map[pol])
    ax.set_ylabel("Energy [J]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Energy Harvested per Slot")

    # (3) Battery SoC
    ax = axes[2]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "S_t" in df.columns:
            _safe_plot(ax, t, df["S_t"], f"{pol} SoC", style_map[pol])
    ax.axhline(y=P.S_target, linestyle="--", color="gray", alpha=0.5, label="SoC target")
    ax.set_ylabel("SoC")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Battery State of Charge")

    # (4) Thermal State
    ax = axes[3]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "T_thermal_C" in df.columns:
            _safe_plot(ax, t, df["T_thermal_C"], f"{pol} Temp", style_map[pol])
        elif "T_next" in df.columns:
            _safe_plot(ax, t, df["T_next"], f"{pol} Temp", style_map[pol])
    ax.axhline(y=P.T_max_C, linestyle="--", color="red", alpha=0.7, label="T_max")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Thermal State")

    # (5) PFD (MPE) Violations
    ax = axes[4]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "pfd_max_Wm2" in df.columns:
            _safe_plot(ax, t, df["pfd_max_Wm2"], f"{pol} PFD", style_map[pol])
    ax.axhline(y=P.S_max_W_per_m2, linestyle="--", color="red", label="PFD_max")
    ax.set_ylabel("PFD [W/m²]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Power Flux Density (MPE Safety)")

    # (6) Interference Violations
    ax = axes[5]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "I_wpt_dBm" in df.columns:
            _safe_plot(ax, t, df["I_wpt_dBm"], f"{pol} I_wpt", style_map[pol])
    ax.axhline(y=P.I_max_dBm, linestyle="--", color="red", label="I_max")
    ax.set_ylabel("Interference [dBm]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Coexistence Interference")

    # (7) Objective Value (E_harv - penalties, higher = better)
    ax = axes[6]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "objective_value" in df.columns:
            _safe_plot(ax, t, df["objective_value"], f"{pol} Obj", style_map[pol])
        elif "J_t" in df.columns:
            # Fallback: negate J_t to get maximization form
            _safe_plot(ax, t, -df["J_t"], f"{pol} Obj", style_map[pol])
    ax.set_ylabel("Objective Value")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Objective: E_harvest - Penalties (higher = better)")

    # (8) SINR (Coexistence Metric)
    ax = axes[7]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "sinr_db" in df.columns:
            _safe_plot(ax, t, df["sinr_db"], f"{pol} SINR", style_map[pol])
    ax.set_ylabel("SINR [dB]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Communication SINR (Signal-to-Interference-plus-Noise Ratio)")

    # (9) Latency
    ax = axes[8]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "latency_ms" in df.columns:
            _safe_plot(ax, t, df["latency_ms"], f"{pol} Latency", style_map[pol])
    ax.axhline(y=P.l_max_ms, linestyle="--", color="red", alpha=0.7, label="l_max")
    ax.set_ylabel("Latency [ms]")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Communication Latency (WPT Duty Cycle Impact)")

    # (10) Action Space: Power & Duty Cycle
    ax = axes[9]
    for pol in policies:
        df = dfs_by_policy[pol]
        t = get_t(df)
        if "P_tx_W" in df.columns:
            _safe_plot(ax, t, df["P_tx_W"], f"{pol} P_tx", style_map[pol])
        if "tau" in df.columns:
            # Scale tau by P_max for visual comparison
            tau_scaled = df["tau"].to_numpy() * P.P_max_W * 0.3  # scale for visibility
            _safe_plot(ax, t, tau_scaled, f"{pol} τ×{0.3*P.P_max_W:.1f}", style_map[pol])
    ax.set_ylabel("Power [W] / Duty (scaled)")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Actions: Transmit Power & Duty Cycle")

    axes[-1].set_xlabel("Time slot")
    fig.suptitle("WPT System: Energy Maximization with Safety Constraints", fontsize=14, y=0.995)

    save_path = out_dir / "fig_timeseries_compare.png"
    fig.savefig(save_path, dpi=150)
    print(f"[saved] {save_path}")

if __name__ == "__main__":
    main()
