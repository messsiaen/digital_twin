# battery.py
import numpy as np

def dcdc_efficiency(Pin_W: float, eta_min: float, eta_max: float, pivot_W: float, steep: float) -> float:
    """Sigmoid-like efficiency vs input power."""
    x = (Pin_W - pivot_W) * steep
    eta = eta_min + (eta_max-eta_min) / (1.0 + np.exp(-x))
    return float(np.clip(eta, eta_min, eta_max))

def ocv_from_soc(soc_pct: float, V_nom: float = 3.7) -> float:
    """Very simple OCV-SOC curve."""
    s = np.clip(soc_pct/100.0, 0.0, 1.0)
    return float(V_nom*(0.9 + 0.15*s))

def step_battery_soc(prev_soc_pct: float, P_chg_W: float, dt_s: float,
                     capacity_mAh: float, coulomb_eff: float = 0.98, Rint_ohm: float = 0.15, V_nom: float = 3.7) -> float:
    """Rint model + coulombic efficiency; P_chg_W positive when charging (net to battery)."""
    V_oc = ocv_from_soc(prev_soc_pct, V_nom)
    # approximate current from power balance V*I = P + I^2*R -> solve quadratic; here small I
    I = P_chg_W / max(V_oc, 1e-3)
    V_term = V_oc - I*Rint_ohm
    I = P_chg_W / max(V_term, 1e-3)
    if I >= 0:
        I_eff = I * coulomb_eff
    else:
        I_eff = I / coulomb_eff
    dAh = I_eff * dt_s / 3600.0
    cap_Ah = capacity_mAh / 1000.0
    new_soc = np.clip(prev_soc_pct + 100.0 * dAh / cap_Ah, 0.0, 100.0)
    return float(new_soc)
