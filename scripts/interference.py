# interference.py
import numpy as np
import math

# -------- common conversions --------
def db_to_lin(x_dB: float) -> float:
    return 10**(x_dB/10.0)
def lin_to_db(x: float) -> float:
    return -1e9 if x<=0 else 10*math.log10(x)

# -------- noise & SINR --------
def sinr_from_components(S_dBm: float, N_dBm: float, I_dBm: float) -> float:
    S = db_to_lin(S_dBm); N = db_to_lin(N_dBm); I = db_to_lin(I_dBm)
    return S / (N + I + 1e-30)

def noise_floor_dBm(bw_MHz: float, NF_dB: float = 6.0, T: float = 290.0) -> float:
    k = 1.38064852e-23
    bw = bw_MHz * 1e6
    N_W = k*T*bw
    return 10*math.log10(N_W)+30 + NF_dB

# -------- WPT near/far-field path model --------
def fspl_dB(d_km: float, f_GHz: float) -> float:
    # 92.45 + 20log10(d_km) + 20log10(f_GHz)
    return 92.45 + 20.0*math.log10(max(d_km, 1e-12)) + 20.0*math.log10(max(f_GHz, 1e-12))

def wpt_path_loss_nearfar_dB(distance_km: float, freq_GHz: float,
                             nf_ref_factor_over_lambda: float = 1.0/(2.0*math.pi),
                             nf_slope_dB_per_dec: float = 60.0,
                             ff_use_fspl: bool = True) -> float:
    """
    Piecewise WPT Tx->Rx path loss:
    - Near-field (r < r_c = lambda/(2π)): PL(r) = PL(r_c) + 60*log10(r/r_c)
    - Far-field (r >= r_c): FSPL(r)
    Continuous at r_c.
    """
    c = 299792458.0
    f_Hz = float(freq_GHz)*1e9
    lam = c / max(f_Hz, 1.0)
    r_c_m = float(nf_ref_factor_over_lambda) * lam
    r_m = max(1e-6, float(distance_km)*1000.0)
    # anchor at r_c
    PL_rc = fspl_dB(r_c_m/1000.0, freq_GHz) if ff_use_fspl else 20.0*math.log10(4.0*math.pi*r_c_m/lam) + 0.0
    if r_m < r_c_m:
        return float(PL_rc + nf_slope_dB_per_dec * math.log10(r_m / r_c_m))
    else:
        return float(fspl_dB(r_m/1000.0, freq_GHz))

def wpt_interference_dBm(P_wpt_tx_dBm: float, path_loss_dB: float,
                         iso_dB: float, rx_filter_rej_dB: float, coupling_dB: float = 0.0) -> float:
    """WPT Tx -> Comm Rx in-band interference after isolation & filtering."""
    return P_wpt_tx_dBm - path_loss_dB - iso_dB - rx_filter_rej_dB + coupling_dB

# -------- EVM / BLER --------
def evm_from_sinr_linear(sinr_lin: float, evm_phase_pct: float = 0.0) -> float:
    evm_noise = 100.0/np.sqrt(max(sinr_lin, 1e-9))
    evm = np.sqrt(evm_noise**2 + evm_phase_pct**2)
    return float(evm)

def bler_from_sinr(sinr_dB: float, th_dB: float = 3.0, slope: float = -1.2) -> float:
    x = (sinr_dB - th_dB) * slope
    p = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(p, 0.0, 1.0))

def residual_bler_with_harq(bler_raw: float, max_tx: int) -> float:
    return float(np.clip(bler_raw ** max_tx, 0.0, 1.0))

# -------- Effective SNR mixers --------
def sinr_effective_capacity_avg(SNR_on_lin: np.ndarray, SNR_off_lin: np.ndarray, duty: float) -> np.ndarray:
    # capacity-average then invert: C_eq = s*log2(1+S_on)+(1-s)*log2(1+S_off); S_eq = 2^C_eq - 1
    C_on  = np.log2(1.0 + np.maximum(SNR_on_lin,  1e-12))
    C_off = np.log2(1.0 + np.maximum(SNR_off_lin, 1e-12))
    C_eq  = duty*C_on + (1.0 - duty)*C_off
    return np.maximum(2.0**C_eq - 1.0, 1e-12)

def sinr_effective_eesm(SNR_on_lin: np.ndarray, SNR_off_lin: np.ndarray, duty: float, beta_lin: float) -> np.ndarray:
    """
    Exponential Effective SNR Mapping (EESM):
      gamma_eff = -beta * ln( s*exp(-gamma_on/beta) + (1-s)*exp(-gamma_off/beta) )
    beta is in *linear* units (convert from dB if user gives dB).
    """
    g_on  = np.maximum(SNR_on_lin,  1e-12)
    g_off = np.maximum(SNR_off_lin, 1e-12)
    t = duty*np.exp(-g_on/beta_lin) + (1.0 - duty)*np.exp(-g_off/beta_lin)
    return np.maximum(-beta_lin * np.log(np.maximum(t, 1e-300)), 1e-12)
