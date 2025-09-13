
"""powercast.py
Rectenna/Powercast-like efficiency curve utilities and loader.
- load_powercast_curve: read CSV [Pin_dBm, efficiency(0-1)], synthesize if missing.
- get_efficiency_interpolator: returns function Pin_dBm -> efficiency.
"""

import os, warnings
import numpy as np
import pandas as pd

def load_powercast_curve(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not {"Pin_dBm","efficiency"}.issubset(df.columns):
            warnings.warn("Powercast CSV missing required columns; synthesizing curve.")
            return synth_powercast_curve()
        return df[["Pin_dBm","efficiency"]].copy()
    return synth_powercast_curve()

def synth_powercast_curve() -> pd.DataFrame:
    pin_dbm = np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    eff =     np.array([0.11,0.26,0.48,0.65,0.70,0.62,0.32,0.10,0.02])
    return pd.DataFrame({"Pin_dBm": pin_dbm, "efficiency": eff})

def get_efficiency_interpolator(curve_df: pd.DataFrame):
    pins = np.array(curve_df["Pin_dBm"].values, dtype=float)
    effs = np.array(curve_df["efficiency"].values, dtype=float)
    def interp(pin_dbm):
        return np.interp(pin_dbm, pins, effs, left=effs[0], right=effs[-1])
    return interp
