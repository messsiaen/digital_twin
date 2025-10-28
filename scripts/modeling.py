from __future__ import annotations
import math
from typing import Dict
import numpy as np
from params import SimParams

# Rain attenuation synthesizer
def synth_rain_attenuation(P: SimParams, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def ar1(m: int, std: float, rho_: float) -> np.ndarray:
        zstd = std * math.sqrt(max(1e-6, 1 - rho_**2))
        z = rng.normal(0.0, zstd, size=m)
        y = np.zeros(m, dtype=float)
        for i in range(1, m):
            y[i] = rho_ * y[i-1] + z[i]
        return y

    def smoothstep01(t: np.ndarray) -> np.ndarray:
        return t*t*(3 - 2*t)

    def trapezoid(duration: int, top_db: float, jitter_std: float, edge_frac: float) -> np.ndarray:
        rise = max(1, int(duration * edge_frac))
        fall = max(1, int(duration * edge_frac))
        plat = max(0, duration - rise - fall)
        seg = np.zeros(duration, dtype=float)
        t = np.linspace(0, 1, rise, endpoint=True)
        seg[:rise] = top_db * smoothstep01(t)
        seg[rise:rise+plat] = top_db
        t = np.linspace(0, 1, fall, endpoint=True)
        seg[rise+plat:] = top_db * (1.0 - smoothstep01(t))
        seg += ar1(duration, jitter_std, P.rain_jitter_rho)
        seg = np.clip(seg, 0.0, None)
        k = max(3, int(duration * 0.02));  k += (k % 2 == 0)
        kernel = np.ones(k, dtype=float) / k
        seg = np.convolve(seg, kernel, mode="same")
        return seg

    n1 = max(20, int(n * P.rain_trap1_dur_frac))
    n2 = max(20, int(n * P.rain_trap2_dur_frac))
    gap = 10
    start1 = rng.integers(0, max(1, n - n1 - n2 - gap))
    end1 = start1 + n1
    left_space  = start1 - gap - n2
    right_space = (n - end1 - gap - n2)
    if max(left_space, right_space) <= 0:
        start2 = min(n - n2, end1 + gap)
    else:
        start2 = rng.integers(0, start1 - gap - n2 + 1) if (rng.random() < 0.5 and left_space > 0) \
                 else rng.integers(end1 + gap, n - n2 + 1)
    end2 = start2 + n2

    rain = np.zeros(n, dtype=float)
    rain[start1:end1] = trapezoid(n1, P.rain_trap1_top_db, P.rain_trap1_jitter_std, P.rain_edge_frac)
    rain[start2:end2] = trapezoid(n2, P.rain_trap2_top_db, P.rain_trap2_jitter_std, P.rain_edge_frac)
    rain += rng.normal(0.0, P.rain_baseline_noise_std, size=n)
    rain += rng.normal(0.0, P.rain_global_bias_std)
    rain = np.clip(rain, 0.0, None)
    return rain

# Unit conversion helpers
def lin_from_dBm(p_dbm: float) -> float:
    return 1e-3 * (10.0 ** (p_dbm / 10.0))

def dBm_from_lin(p_w: float) -> float:
    if p_w <= 0: return -300.0
    return 10.0 * math.log10(p_w / 1e-3)

def lin_from_dB(val_db: float) -> float:
    return 10.0 ** (val_db / 10.0)

def dB_from_lin(val_lin: float) -> float:
    if val_lin <= 0: return -300.0
    return 10.0 * math.log10(val_lin)

# Path loss & atmospheric attenuation
def fspl_db(d_m: float, f_hz: float) -> float:
    """Free-space path loss [dB]"""
    c = 299792458.0
    if d_m <= 0 or f_hz <= 0:
        return 0.0
    return 20.0 * math.log10(4.0 * math.pi * d_m * f_hz / c)

def rain_attenuation_db(_f_hz: float, _rain_rate_mm_hr: float, _d_eff_m: float) -> float:
    """ITU-R P.838 simplified rain attenuation"""
    return 0.0  # placeholder; actual rain from synth_rain_attenuation

def gas_attenuation_db(f_hz: float, P_kPa: float, T_C: float, rho_v_g_m3: float, d_eff_m: float) -> float:
    """ITU-R P.676 simplified gaseous attenuation (O2 + H2O)"""
    f_GHz = f_hz / 1e9
    # Oxygen absorption (peaks ~60 GHz, ~120 GHz)
    gamma_o = 0.0
    if f_GHz < 10:
        gamma_o = 7.2e-3 * (P_kPa / 101.3) * (f_GHz**2) / ((f_GHz**2 + 0.34) * (f_GHz**2 + 0.8))
    else:
        gamma_o = 0.01 * f_GHz**0.5  # rough approximation

    # Water vapor absorption (peak ~22 GHz, ~183 GHz)
    gamma_w = 0.05 * (rho_v_g_m3 / 7.5) * (f_GHz**2) / ((f_GHz - 22.2)**2 + 2.5)

    gamma_total_dB_km = (gamma_o + gamma_w)
    return gamma_total_dB_km * (d_eff_m / 1000.0)

def cloud_attenuation_db(f_hz: float, lwc_g_m3: float, d_eff_m: float) -> float:
    """ITU-R P.840 cloud attenuation"""
    f_GHz = f_hz / 1e9
    K_l = 0.4 * f_GHz
    gamma_c_dB_km = K_l * lwc_g_m3
    return gamma_c_dB_km * (d_eff_m / 1000.0)

def total_path_loss_db(P: SimParams, d_m: float, rain_db: float, f_hz: float) -> float:
    """Total path loss: FSPL + rain + gas + cloud"""
    L_fspl = fspl_db(d_m, f_hz)

    rho_v = (P.atm_humidity_percent / 100.0) * 17.3 * math.exp(-4000 / (P.atm_temp_C + 273.15))
    A_gas = gas_attenuation_db(f_hz, P.atm_pressure_kPa, P.atm_temp_C, rho_v, P.d_eff_m)
    A_cloud = cloud_attenuation_db(f_hz, P.cloud_lwc_g_m3, P.d_eff_m)
    A_rain = max(0.0, rain_db)
    return L_fspl + A_rain + A_gas + A_cloud

# Antenna gain with beamwidth & polarization
def beam_gain_from_width(beamwidth_deg: float, pointing_error_deg: float = 0.0) -> float:
    """Antenna gain from 3dB beamwidth with pointing loss"""
    if beamwidth_deg <= 0:
        return 1.0
    G_max_dB = 10.0 * math.log10(30000.0 / max(1.0, beamwidth_deg**2))
    G_max_lin = lin_from_dB(G_max_dB)

    if pointing_error_deg > 0:
        loss_factor = math.exp(-2.77 * (pointing_error_deg / beamwidth_deg)**2)
        return G_max_lin * loss_factor
    return G_max_lin

def polarization_mismatch_loss(pol_tx: str, pol_rx: str) -> float:
    """Polarization mismatch factor χ ∈ [0,1]"""
    if pol_tx == pol_rx:
        return 1.0
    if (pol_tx == "H" and pol_rx == "V") or (pol_tx == "V" and pol_rx == "H"):
        return 0.01
    if (pol_tx in ["RHCP", "LHCP"]) and (pol_rx in ["RHCP", "LHCP"]):
        if pol_tx != pol_rx:
            return 0.01
        return 1.0
    if (pol_tx in ["H", "V"]) and (pol_rx in ["RHCP", "LHCP"]):
        return 0.5
    if (pol_tx in ["RHCP", "LHCP"]) and (pol_rx in ["H", "V"]):
        return 0.5
    return 0.5

def ris_coherent_gain(P: SimParams, phase_optimize: bool = False) -> float:
    """RIS coherent gain (N^2 when optimally phased)"""
    if not P.ris_enable:
        return 1.0

    if phase_optimize:
        return min(P.ris_gain_max, float(P.ris_n_elements)**1.5)
    else:
        return float(P.ris_n_elements)**0.5

def composite_gain_lin(P: SimParams, d_m: float, rain_db: float,
                       subband_idx: int = 0, beamwidth_idx: int = 0,
                       pol_tx: str = "V", ris_phase_opt: bool = False) -> float:
    """Total composite gain: G_TX * G_RX * G_RIS * χ * G_path"""
    f_wpt = P.f_wpt_hz + (P.wpt_subbands_MHz[subband_idx] * 1e6)
    L_total_dB = total_path_loss_db(P, d_m, rain_db, f_wpt)
    G_path = lin_from_dB(-L_total_dB)
    beamwidth = P.beam_widths_deg[beamwidth_idx]
    G_beam = beam_gain_from_width(beamwidth, pointing_error_deg=0.0)
    G_TX = P.g_tx_base_lin * G_beam
    G_RX = P.g_rx_base_lin
    chi = polarization_mismatch_loss(pol_tx, P.pol_rx_default)
    G_RIS = ris_coherent_gain(P, ris_phase_opt)
    return G_TX * G_RX * G_RIS * chi * G_path

# Waveform-dependent rectifier gain
def waveform_rectifier_gain(waveform_type: str, multisine_weights: np.ndarray = None) -> float:
    """Waveform-induced rectifier gain κ_wf"""
    if waveform_type == "CW":
        return 1.0

    if waveform_type == "Multisine":
        K = len(multisine_weights) if multisine_weights is not None else 4
        return 1.0 + 0.15 * math.sqrt(K)
    return 1.0

# Rectifier efficiency (temperature-dependent)
def wpt_efficiency(P: SimParams, P_rx_W: float, T_C: float = 25.0,
                   kappa_wf: float = 1.0) -> float:
    """Rectifier efficiency with temperature derating"""
    if P_rx_W <= 0:
        return 0.0

    # Temperature derating
    delta_T = max(0.0, T_C - P.T_amb_C)
    eta_max_T = P.wpt_eta_max * (1.0 - P.wpt_temp_derating * delta_T)
    eta_max_T = max(0.1, min(eta_max_T, 0.95))

    # Nonlinear efficiency curve
    x = max(0.0, P_rx_W / max(1e-20, P.wpt_eff_p0_W))
    eta_base = eta_max_T * (x**P.wpt_eff_k) / (1.0 + x**P.wpt_eff_k)

    # Waveform gain
    eta = eta_base * kappa_wf
    return float(max(0.0, min(eta, 0.95)))

def harvested_energy_J(P: SimParams, P_rx_W: float, tau: float, T_C: float = 25.0,
                       kappa_wf: float = 1.0) -> float:
    """Energy harvested in one slot [J]."""
    eta = wpt_efficiency(P, P_rx_W, T_C, kappa_wf)
    return tau * P.delta_s * eta * P_rx_W

# ================ Battery & Thermal State Updates ================
def battery_step(P: SimParams, S_t: float, E_harv_J: float, E_consumed_J: float) -> float:
    """Update battery SoC."""
    S_next = S_t + (E_harv_J - E_consumed_J) / P.E_bat_J
    return float(min(P.S_max, max(P.S_min, S_next)))

def thermal_step(P: SimParams, T_t: float, tau: float, P_rx_W: float) -> float:
    """
    First-order RC thermal model:
    T_{t+1} = a * T_t + b * (τ * P_rx) + c
    where c = T_amb * (1 - a)
    """
    T_next = P.thermal_a * T_t + P.thermal_b * (tau * P_rx_W) + P.thermal_c
    return float(max(P.T_amb_C, min(T_next, 100.0)))  # cap at 100°C

# ================ Safety Metrics ================
def hinge(x: float) -> float:
    """Hinge function: max(0, x)"""
    return max(0.0, x)

def pfd_at_point(P: SimParams, P_tx_W: float, u: int,
                 beamwidth_idx: int, r_m: float) -> float:
    """
    Power flux density at protected point m [W/m^2].
    S_m = u * P_tx * σ_m(θ, β, π) / r_m^2
    Simplified: σ_m ≈ G_beam / 4π
    """
    if u == 0 or P_tx_W <= 0:
        return 0.0

    beamwidth = P.beam_widths_deg[beamwidth_idx]
    G_beam = beam_gain_from_width(beamwidth)
    sigma_m = G_beam / (4.0 * math.pi)

    return u * P_tx_W * sigma_m / max(0.1, r_m**2)

def interference_at_victim(P: SimParams, P_tx_W: float, u: int, subband_idx: int,
                            waveform_type: str, beamwidth_idx: int) -> float:
    """
    Coexistence interference at victim receiver [W].
    I_wpt = u * P_tx * ψ_leak(b, ζ, Δf) * g_r(θ, β, π)
    """
    if u == 0 or P_tx_W <= 0:
        return 0.0

    # Spectral leakage factor (depends on subband and waveform)
    delta_f = abs(P.wpt_subbands_MHz[subband_idx])
    psi_leak = P.kappa_leak * math.exp(-delta_f / 50.0)  # decays with freq separation

    # Spatial coupling (directional gain toward victim)
    beamwidth = P.beam_widths_deg[beamwidth_idx]
    G_beam = beam_gain_from_width(beamwidth)
    g_r = P.g_leak_lin * G_beam / max(1.0, G_beam**0.5)  # reduced by directionality

    return u * P_tx_W * psi_leak * g_r

# ================ Latency ================
def latency_from_tau(P: SimParams, tau: float) -> float:
    """Latency [ms] = l0 + κ_l * τ"""
    return P.l0_ms + P.kappa_l_ms * float(tau)

# ================ Legacy Interface (for backward compatibility) ================
class PesqInterface:
    """Dummy PESQ interface (PESQ removed per new spec)."""
    def __init__(self, P: SimParams):
        self.P = P

    def predict(self, gamma_pre_db: float, p_wpt_dbm: float,
                delta_f_MHz: float, q_kbps: int, w_level: float) -> float:
        # No PESQ in new spec; return dummy value
        return 3.0

def w_level_from_dbm(P: SimParams, p_wpt_dbm: float) -> float:
    """Discretize WPT power level (legacy)."""
    arr = np.asarray(P.wpt_rx_dbm_levels, dtype=float)
    j = int(np.argmin(np.abs(arr - p_wpt_dbm)))
    return float(j)

def pre_snr_db(S_dBm: float, N0B_dBm: float, I_bg_dBm: float, I_wpt_dBm: float) -> float:
    """Pre-processing SINR [dB] (for coexistence check only)."""
    S = lin_from_dBm(S_dBm)
    N = lin_from_dBm(N0B_dBm)
    Ibg = lin_from_dBm(I_bg_dBm)
    Iw = lin_from_dBm(I_wpt_dBm)
    denom = max(1e-15, N + Ibg + Iw)
    return 10.0 * math.log10(S / denom)

# ================ Main Physics Step ================
def step_physics(P: SimParams, state: Dict, vis_t: int, rain_db_t: float,
                 u: int, P_tx_W: float, tau: float,
                 subband_idx: int = 0, beamwidth_idx: int = 0,
                 pol_tx: str = "V", waveform_type: str = "CW",
                 ris_phase_opt: bool = False,
                 prev_action: Dict = None,
                 pesq: "PesqInterface" = None) -> Dict:
    """
    NEW physics step with expanded action space and safety penalties.

    Decision variables (per spec):
    - u: WPT on/off
    - tau: duty cycle
    - P_tx_W: transmit power
    - subband_idx: frequency subband
    - beamwidth_idx: beam shape
    - pol_tx: polarization
    - waveform_type: waveform family
    - ris_phase_opt: RIS phase optimization flag

    Returns dict with:
    - States: S_next, T_next
    - Energy: E_harv_J
    - Safety: PFD violations, interference violations, thermal violations
    - Objective: energy minus penalties
    """

    # ========== State Extraction ==========
    S_t = float(state.get("S", P.S0))
    T_t = float(state.get("T", P.T0_C))

    # ========== Action Constraints ==========
    tau = float(min(max(0.0, tau), 1.0, P.latency_tau_max()))
    P_tx_W = float(min(max(0.0, P_tx_W), P.P_max_W))
    u = int(1 if (vis_t and P_tx_W > 0 and tau > 0) else 0)

    # Slew rate limits (if prev_action provided)
    if prev_action is not None:
        P_prev = float(prev_action.get("P_tx_W", 0.0))
        tau_prev = float(prev_action.get("tau", 0.0))

        if abs(P_tx_W - P_prev) > P.max_delta_P_W:
            P_tx_W = P_prev + math.copysign(P.max_delta_P_W, P_tx_W - P_prev)

        if abs(tau - tau_prev) > P.max_delta_tau:
            tau = tau_prev + math.copysign(P.max_delta_tau, tau - tau_prev)

    # ========== Channel & Reception ==========
    G_composite = composite_gain_lin(P, P.d_m, rain_db_t, subband_idx,
                                      beamwidth_idx, pol_tx, ris_phase_opt)
    P_rx_W = u * P_tx_W * G_composite
    p_wpt_dbm = dBm_from_lin(P_rx_W) if P_rx_W > 0 else -300.0

    # ========== Waveform Gain ==========
    multisine_weights = np.ones(P.multisine_tones) / P.multisine_tones
    kappa_wf = waveform_rectifier_gain(waveform_type, multisine_weights)

    # ========== Energy Harvesting ==========
    E_harv_J = harvested_energy_J(P, P_rx_W, tau, T_t, kappa_wf)

    # ========== Energy Consumption ==========
    E_base_J = P.delta_s * P.P_base_W
    E_comm_J = P.delta_s * P.P_comm_W * max(0.0, 1.0 - tau)
    E_consumed_J = E_base_J + E_comm_J

    # ========== State Updates ==========
    S_next = battery_step(P, S_t, E_harv_J, E_consumed_J)
    T_next = thermal_step(P, T_t, tau, P_rx_W)

    # ========== Safety Metrics ==========
    # (S1) MPE / PFD with soft penalty
    pfd_values = []
    pfd_penalties_list = []
    for r_m in P.protected_points_r_m:
        pfd_m = pfd_at_point(P, P_tx_W, u, beamwidth_idx, r_m)
        pfd_values.append(pfd_m)

        # Soft penalty: no penalty below optimal, linear above optimal
        # optimal = max - 10, coefficient = 0.6
        pfd_optimal = P.S_max_W_per_m2 - 10.0
        if pfd_m > pfd_optimal:
            penalty = P.lambda_mpe * abs(pfd_m - pfd_optimal)
        else:
            penalty = 0.0
        pfd_penalties_list.append(penalty)

    pfd_penalty = sum(pfd_penalties_list)
    pfd_max_Wm2 = max(pfd_values) if pfd_values else 0.0

    # (S2) Coexistence interference with soft penalty
    I_wpt_W = interference_at_victim(P, P_tx_W, u, subband_idx, waveform_type, beamwidth_idx)
    I_wpt_dBm = dBm_from_lin(I_wpt_W) if I_wpt_W > 0 else -300.0

    # Soft penalty: no penalty below optimal, linear above optimal
    # optimal = max - 50, coefficient = 0.08
    int_optimal = P.I_max_dBm - 50.0
    if I_wpt_dBm > int_optimal:
        int_penalty = P.lambda_int * abs(I_wpt_dBm - int_optimal)
    else:
        int_penalty = 0.0

    # (S3) Thermal with soft penalty
    # Soft penalty: no penalty below optimal, linear above optimal
    # optimal = max - 30, coefficient = 0.1
    th_optimal = P.T_max_C - 30.0
    if T_next > th_optimal:
        th_penalty = P.lambda_th * abs(T_next - th_optimal)
    else:
        th_penalty = 0.0

    # ========== Objective (NEW: maximize energy - safety risks) ==========
    # Total penalties (already computed above with soft penalty logic)
    total_penalty = pfd_penalty + int_penalty + th_penalty

    # J_t for minimization (used by optimizer)
    J_t = -E_harv_J + total_penalty

    # Objective value for display (maximization perspective)
    # Limit to >= 0 to avoid negative objective
    objective_value = max(0.0, E_harv_J - total_penalty)

    # ========== Constraint Checks ==========
    pfd_ok = all(v <= P.S_max_W_per_m2 + 1e-6 for v in pfd_values)
    int_ok = (I_wpt_dBm <= P.I_max_dBm + 1e-6)
    th_ok = (T_next <= P.T_max_C + 1e-6)
    latency_ok = (latency_from_tau(P, tau) <= P.l_max_ms + 1e-6)
    feas_all = (u == 0) or (pfd_ok and int_ok and th_ok and latency_ok)

    # ========== SINR Calculation (for coexistence check) ==========
    sinr_db = pre_snr_db(P.S_sig_dBm, P.N0B_dBm, P.I_bg_dBm, I_wpt_dBm)

    # ========== Return Results ==========
    return {
        # States
        "S_t": S_t,
        "S_next": S_next,
        "T_t": T_t,
        "T_next": T_next,

        # Actions
        "u": u,
        "P_tx_W": P_tx_W,
        "tau": tau,
        "subband_idx": subband_idx,
        "beamwidth_idx": beamwidth_idx,
        "pol_tx": pol_tx,
        "waveform_type": waveform_type,
        "ris_phase_opt": ris_phase_opt,

        # Channel
        "G_composite": G_composite,
        "P_rx_W": P_rx_W,
        "p_wpt_dbm": p_wpt_dbm,
        "kappa_wf": kappa_wf,

        # Energy
        "E_harv_J": E_harv_J,
        "E_consumed_J": E_consumed_J,

        # Safety metrics
        "pfd_max_Wm2": pfd_max_Wm2,
        "pfd_penalty": pfd_penalty,
        "I_wpt_dBm": I_wpt_dBm,
        "int_penalty": int_penalty,
        "T_thermal_C": T_next,
        "th_penalty": th_penalty,

        # Constraints
        "pfd_ok": pfd_ok,
        "int_ok": int_ok,
        "th_ok": th_ok,
        "latency_ok": latency_ok,
        "feas_all": feas_all,

        # Objective
        "J_t": J_t,
        "objective_value": objective_value,  # for display: E_harv - penalties

        # Coexistence
        "sinr_db": sinr_db,  # SINR for comm link

        # Other
        "rain_db": rain_db_t,
        "vis": vis_t,
        "latency_ms": latency_from_tau(P, tau),
    }
