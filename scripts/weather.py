# weather.py
"""
P.618-style slant-path rain attenuation with RSS composition:
A_tot = A_gas + sqrt( A_rain^2 + A_scint^2 )
- Rain: P.838 (k, alpha) * P.618 slant geometry with horizontal reduction r
- Gas:  simple P.676-like slope (placeholder)
- Scintillation: simple P.618-like RMS model (frequency & elevation dependent)
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np

# ------------------------------
# P.838: k, alpha lookup/approx
# ------------------------------
def k_alpha_p838(freq_GHz: float, pol: str = "H") -> Tuple[float, float]:
    pol = "H" if str(pol).upper().startswith("H") else "V"
    table = {
        5.8:  (9.2e-4, 0.85, 7.2e-4, 0.87),
        10.0: (3.3e-3, 0.90, 2.8e-3, 0.91),
        12.0: (4.8e-3, 0.91, 4.1e-3, 0.92),
        20.0: (1.5e-2, 0.95, 1.3e-2, 0.96),
        30.0: (3.7e-2, 0.98, 3.2e-2, 0.99),
        40.0: (7.2e-2, 1.00, 6.2e-2, 1.01),
    }
    key = min(table.keys(), key=lambda f: abs(f - freq_GHz))
    (kH, aH, kV, aV) = table[key]
    return (kH, aH) if pol == "H" else (kV, aV)

def specific_attenuation_gamma(freq_GHz: float, rain_rate_mmph: float, pol: str = "H") -> float:
    k, a = k_alpha_p838(freq_GHz, pol)
    R = max(0.0, float(rain_rate_mmph))
    return float(k * (R ** a))  # dB/km

# ----------------------------------------
# P.618: slant geometry & reduction factor
# ----------------------------------------
def _deg2rad(x: float) -> float:
    return math.radians(float(x))

def effective_slant_path_km(elevation_deg: float, h_station_km: float = 0.1, h_rain_km: Optional[float] = None) -> Tuple[float, float]:
    theta = max(5.0, float(elevation_deg))
    th = _deg2rad(theta)
    hr = 5.0 if (h_rain_km is None) else float(h_rain_km)  # coarse default
    hs = float(h_station_km)
    if hr <= hs:
        return 0.0, 0.0
    Ls = (hr - hs) / max(math.sin(th), 1e-6)
    Lg = Ls * math.cos(th)
    return Ls, Lg

def horizontal_reduction_factor_r(Lg_km: float, gamma_R: float, freq_GHz: float) -> float:
    Lg = max(0.0, float(Lg_km))
    gR = max(0.0, float(gamma_R))
    f = max(0.1, float(freq_GHz))
    denom = 1.0 + 0.78 * math.sqrt(max(0.0, Lg * gR / f)) - 0.38 * (1.0 - math.exp(-2.0 * Lg))
    return float(1.0 / max(denom, 1e-6))

def rain_attenuation_p618_instant(rain_rate_mmph: float, freq_GHz: float, elevation_deg: float,
                                  pol: str = "H", h_station_km: float = 0.1, h_rain_km: Optional[float] = None,
                                  use_itu_libs: bool = False) -> float:
    # (optional) try 'itur' when available for site-dependent parameters; fallback to this approximation
    if use_itu_libs:
        try:
            import itur  # type: ignore
            # For instant-R mode we keep local formula; add percentile mode separately when needed.
        except Exception:
            pass
    gamma_R = specific_attenuation_gamma(freq_GHz, rain_rate_mmph, pol=pol)
    Ls, Lg = effective_slant_path_km(elevation_deg, h_station_km=h_station_km, h_rain_km=h_rain_km)
    r = horizontal_reduction_factor_r(Lg, gamma_R, freq_GHz)
    return float(gamma_R * Ls * r)

# -----------------------------------------
# Gas / Scintillation (lightweight models)
# -----------------------------------------
def gas_specific_attenuation_p676_simple(freq_GHz: float) -> float:
    f = float(freq_GHz)
    if f < 10.0:
        return 0.005
    elif f < 20.0:
        return 0.015
    elif f < 30.0:
        return 0.035
    else:
        return 0.06  # dB/km

def cloud_fog_attenuation_p840_simple(freq_GHz: float, elevation_deg: float) -> float:
    return 0.05 if freq_GHz >= 20.0 else 0.01  # dB lumped small term

def scintillation_rms_p618_like_dB(freq_GHz: float, elevation_deg: float) -> float:
    """
    Simple RMS scintillation amplitude in dB, increasing with frequency and decreasing with elevation.
    Tuned to ~0.5 dB @ 20 GHz, 10 deg; gently limited to [0.1, 2.0] dB.
    """
    f = max(1.0, float(freq_GHz))
    el = max(5.0, float(elevation_deg))
    sigma = 0.5 * (f / 20.0) ** 0.7 * (10.0 / el) ** 0.3
    return float(min(2.0, max(0.1, sigma)))

# ------------------------------
# Public API
# ------------------------------
def total_attenuation_db(rain_mmph: float, freq_GHz: float, elevation_deg: float,
                         use_itu_libs: bool = False, add_gas: bool = True, add_cloud: bool = True, add_scint: bool = True,
                         *, pol: str = "H", h_station_km: float = 0.1, h_rain_km: Optional[float] = None,
                         use_p618: bool = True, use_rss_composition: bool = True) -> float:
    """
    Total slant-path attenuation (dB) for instantaneous rain rate R:
      A_tot = A_gas + sqrt( A_rain^2 + A_scint^2 )
    """
    # Rain
    if use_p618:
        A_rain = rain_attenuation_p618_instant(rain_mmph, freq_GHz, elevation_deg, pol, h_station_km, h_rain_km, use_itu_libs)
    else:
        th = _deg2rad(max(5.0, elevation_deg))
        gamma_R = specific_attenuation_gamma(freq_GHz, rain_mmph, pol=pol)
        L_eff_simple = 2.0 / max(math.sin(th), 1e-6)  # km
        A_rain = float(gamma_R * L_eff_simple)

    # Gas (as slant path)
    A_gas = gas_specific_attenuation_p676_simple(freq_GHz) * (1.0 / max(math.sin(_deg2rad(max(5.0, elevation_deg))), 1e-6)) if add_gas else 0.0
    # Cloud/fog (small lumped)
    A_cloud = cloud_fog_attenuation_p840_simple(freq_GHz, elevation_deg) if add_cloud else 0.0
    # Scintillation RMS
    A_scint = scintillation_rms_p618_like_dB(freq_GHz, elevation_deg) if add_scint else 0.0

    if use_rss_composition:
        A_tot = A_gas + math.sqrt(A_rain**2 + A_scint**2) + A_cloud
    else:
        A_tot = A_gas + A_rain + A_scint + A_cloud
    return float(A_tot)
