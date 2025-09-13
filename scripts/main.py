# main.py
"""
Core 24h simulation packed into `run_simulation(P, seed, out_dir, rain_override=None, storm=None)`.
Other experiment scripts import this function.

Dependencies:
  params.py (SimParams), weather.py, interference.py, battery.py,
  emodel.py, traffic.py, powercast.py, optimizer.py
"""

from __future__ import annotations
import os, math, csv
from typing import Dict, Tuple, List, Optional
import numpy as np

from params import SimParams
from weather import total_attenuation_db
from interference import (
    noise_floor_dBm, wpt_interference_dBm, fspl_dB, wpt_path_loss_nearfar_dB
)
from powercast import load_powercast_curve, get_efficiency_interpolator
from battery import dcdc_efficiency, step_battery_soc
from emodel import mos_from_emodel
from traffic import BitQueue, demand_from_bitrate_kbps

from optimizer import (
    choose_greedy_mos, choose_energy_first, choose_twostage, choose_random, maybe_explore,
    choose_rfwf, choose_dp2_modeplan, choose_dp2_cvar_rfwf, choose_sw_idle, choose_sw_twostage,
    _bler_mixed, _snr_eff_capacity_avg, _arrival_kbps, _impair_from_table
)

# ---------------------------
# Environment synthesis
# ---------------------------
def synth_rain_24h_spiky(n: int = 5760, seed: int = 1) -> np.ndarray:
    """Frequent spiky rain with uniform base; tuned to show rich dynamics with larger spikes up to 45mm/h."""
    rng = np.random.default_rng(seed)
    
    # Start with uniform rain as base
    rain = synth_rain_24h_uniform(n, seed=seed+1000)
    
    # Main bursts - reduced size and frequency
    k_main = rng.integers(60, 90)  # Reduced from 80-120 to 60-90
    centers = rng.integers(0, n, size=k_main)
    peaks   = rng.uniform(5.0, 20.0, size=k_main)  # Reduced from 8.0-30.0 to 5.0-20.0
    widths  = rng.integers(6, 18,  size=k_main)    # Reduced from 8-25 to 6-18
    for c, p, w in zip(centers, peaks, widths):
        left = max(0, c - w); right = min(n, c + w + 1)
        x = np.arange(left, right) - c
        shape = np.exp(-np.abs(x) / max(1, w/3))
        rain[left:right] += p * shape

    # Micro-showerlets - reduced size
    k_micro = rng.integers(150, 250)  # Reduced from 200-350 to 150-250
    centers2 = rng.integers(0, n, size=k_micro)
    peaks2   = rng.uniform(0.5, 5.0, size=k_micro)  # Reduced from 1.0-8.0 to 0.5-5.0
    widths2  = rng.integers(1, 6, size=k_micro)     # Reduced from 2-8 to 1-6
    for c, p, w in zip(centers2, peaks2, widths2):
        w_eff = max(1, int(w))
        left = max(0, c - w_eff); right = min(n, c + w_eff + 1)
        x = np.arange(left, right) - c
        tri = (w_eff - np.abs(x)) / w_eff
        tri[tri < 0] = 0.0
        rain[left:right] += p * tri

    # Additional random spikes - reduced size
    k_extra = rng.integers(30, 70)  # Reduced from 50-100 to 30-70
    centers3 = rng.integers(0, n, size=k_extra)
    peaks3   = rng.uniform(3.0, 15.0, size=k_extra)  # Reduced from 5.0-25.0 to 3.0-15.0
    widths3  = rng.integers(3, 10, size=k_extra)     # Reduced from 5-15 to 3-10
    for c, p, w in zip(centers3, peaks3, widths3):
        left = max(0, c - w); right = min(n, c + w + 1)
        x = np.arange(left, right) - c
        # Use a more gradual decay for smoother spikes
        shape = np.exp(-np.abs(x) / max(1, w/2))
        rain[left:right] += p * shape

    # Enhanced drizzle + jitter
    rain += rng.gamma(shape=0.6, scale=0.2, size=n)  # Increased from 0.45, 0.15
    rain += rng.normal(0.0, 0.15, size=n)            # Increased from 0.12

    # Clip/scale - reduced maximum
    p99 = np.percentile(rain, 99)
    if p99 > 35.0:  # Reduced from 50.0 to 35.0
        rain *= (35.0 / p99)
    return np.clip(rain, 0.0, None)

def synth_rain_24h_uniform(n: int = 5760, seed: int = 10) -> np.ndarray:
    """Uniform rain distribution with sustained periods and natural transitions."""
    rng = np.random.default_rng(seed)
    rain = np.zeros(n, dtype=float)
    
    # Create 3-4 sustained rain periods
    num_periods = rng.integers(3, 5)
    base_durations = rng.integers(120, 300, size=num_periods)  # 2-5 hours each
    # Randomly multiply duration by 2-4x
    duration_multipliers = rng.uniform(2.0, 4.0, size=num_periods)
    period_durations = (base_durations * duration_multipliers).astype(int)
    period_starts = rng.integers(0, n - max(period_durations), size=num_periods)
    period_intensities = rng.uniform(12.0, 30.0, size=num_periods)  # Increased base intensity 12-30 mm/h
    
    for start, duration, intensity in zip(period_starts, period_durations, period_intensities):
        end = min(start + duration, n)
        period_length = end - start
        
        # Create longer, more natural transitions
        transition_length = min(60, period_length // 3)  # 60 minutes or 1/3 of period
        
        # Main sustained period with natural fluctuations
        for i in range(period_length):
            pos = start + i
            
            # Add natural random fluctuations with varying intensity
            base_fluctuation = intensity * 0.15  # ±15% base fluctuation
            fluctuation = rng.normal(0, base_fluctuation)
            
            # Add occasional larger variations for more natural feel
            if rng.random() < 0.1:  # 10% chance of larger variation
                fluctuation += rng.normal(0, intensity * 0.2)
            
            # Smooth, natural transitions at edges using sigmoid-like curves
            if i < transition_length:
                # Rising transition with smooth curve
                t = i / transition_length
                # Use a smooth S-curve instead of linear
                factor = 0.5 * (1 + np.tanh(4 * (t - 0.5)))
                rain[pos] = intensity * factor + fluctuation
            elif i >= period_length - transition_length:
                # Falling transition with smooth curve
                t = (period_length - i) / transition_length
                factor = 0.5 * (1 + np.tanh(4 * (t - 0.5)))
                rain[pos] = intensity * factor + fluctuation
            else:
                # Sustained period with natural variations
                rain[pos] = intensity + fluctuation
    
    # Add natural background drizzle between periods
    drizzle_mask = rain < 1.5
    drizzle_intensity = rng.uniform(0.2, 1.5, size=n)
    rain[drizzle_mask] += drizzle_intensity[drizzle_mask]
    
    # Add natural random noise with varying intensity
    noise_std = 0.2 + 0.1 * (rain / 20.0)  # Noise increases with rain intensity
    noise = rng.normal(0, noise_std, size=n)
    rain += noise
    
    # Only clip negative values, no upper limit for natural variation
    return np.clip(rain, 0.0, None)

def inject_extreme_storm(rain: np.ndarray, start_min: int = 600, duration_min: int = 90,
                         peak_mmph: float = 28.0, shape: str = "tri") -> np.ndarray:
    """Insert an adversarial 90-min storm into rain series."""
    n = len(rain)
    s = max(0, min(n-1, start_min))
    e = max(s+1, min(n, s + duration_min))
    L = e - s
    if shape == "tri":
        x = np.arange(L)
        tri = 1.0 - np.abs((x - (L-1)/2) / ((L-1)/2))
        tri[tri < 0] = 0.0
        pulse = peak_mmph * tri
    else:
        # default raised cosine
        x = np.arange(L)
        pulse = peak_mmph * 0.5*(1 - np.cos(2*np.pi*x/(L-1)))
    out = rain.copy()
    out[s:e] = np.maximum(out[s:e], pulse)
    return out

def synth_temp_24h(n=5760, seed=2, Tmin=9.0, Tmax=30.0) -> np.ndarray:
    """Daily temperature sinusoid with noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/n
    base = (Tmax+Tmin)/2.0 + (Tmax-Tmin)/2.0 * np.sin(2*np.pi*(t-0.25))
    return base + rng.normal(0, 1.2, size=n)

def absolute_humidity_gm3(temp_C: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    """Tetens formula based absolute humidity."""
    T = np.clip(temp_C, -20, 45)
    RH = np.clip(RH_pct, 1, 100)
    es = 6.112 * np.exp(17.67*T/(T+243.5))
    e = es * (RH/100.0)
    return 216.7 * e / (T + 273.15)

def synth_humidity_24h(temp_C: np.ndarray, rain_mmph: np.ndarray, seed=3):
    """Humidity co-varies with diurnal cycle and rain."""
    rng = np.random.default_rng(seed)
    n = len(temp_C); t = np.arange(n)/n
    rh = 70 + 20*np.cos(2*np.pi*(t-0.25)) + rng.normal(0, 4, size=n)
    rh += np.clip(rain_mmph, 0, 20)*0.8
    rh = np.clip(rh, 30, 100)
    ah = absolute_humidity_gm3(temp_C, rh)
    return rh, ah

# ---------------------------
# WPT & link helpers
# ---------------------------
def wpt_chain(P: SimParams, A_wpt_dB: float) -> Tuple[float, float, float]:
    """Return (P_rf_dBm_on, P_dc_W_on, path_loss_dB including A_wpt & pointing)."""
    if P.use_wpt_nearfield_model:
        PL = wpt_path_loss_nearfar_dB(
            P.wpt_distance_km, P.wpt_freq_GHz,
            nf_ref_factor_over_lambda=P.nf_ref_factor_over_lambda,
            nf_slope_dB_per_dec=P.nf_slope_dB_per_dec,
            ff_use_fspl=P.ff_use_fspl
        ) + P.wpt_pointing_loss_db + A_wpt_dB
    else:
        PL = fspl_dB(P.wpt_distance_km, P.wpt_freq_GHz) + P.wpt_pointing_loss_db + A_wpt_dB

    P_rf_dBm_on = P.wpt_tx_power_dbm - PL
    curve = load_powercast_curve(path="")
    eta_rect = get_efficiency_interpolator(curve)
    eta_raw = eta_rect(P_rf_dBm_on)
    gate = 1.0 / (1.0 + np.exp(-(P_rf_dBm_on - P.rect_knee_dbm)/max(1e-3, P.rect_knee_width_db)))
    eta_eff = eta_raw * gate
    P_rf_W_on = 10 ** ((P_rf_dBm_on - 30.0)/10.0)
    eta_dcdc = dcdc_efficiency(P_rf_W_on*eta_eff, P.dcdc_eta_min, P.dcdc_eta_max, P.dcdc_eta_pivot_W, P.dcdc_eta_steep)
    P_dc_W_on = P_rf_W_on * eta_eff * eta_dcdc
    return float(P_rf_dBm_on), float(P_dc_W_on), float(PL)

# ---------------------------
# Core simulation
# ---------------------------
def run_simulation(P: SimParams,
                   seed: int = 2025,
                   out_dir: Optional[str] = None,
                   rain_override: Optional[np.ndarray] = None,
                   storm: Optional[Dict] = None):
    """
    Run 24h simulation and optionally save per-baseline CSVs.

    Args:
      P: SimParams config object (will be read only).
      seed: RNG seed.
      out_dir: if provided, save metrics CSV and return sim dict.
      rain_override: optional array of length 5760 to replace synthesized rain.
      storm: optional dict for adversarial injection, e.g.
             {"start_min":600,"duration_min":90,"peak_mmph":28.0,"shape":"tri"}

    Returns:
      sim dict containing env, attenuations, per-baseline time series, s_bar_hour, etc.
    """
    n = P.total_slots
    rng = np.random.default_rng(seed)

    # ---- Environment ----
    rain = synth_rain_24h_spiky(n, seed=seed) if rain_override is None else np.asarray(rain_override, dtype=float).copy()
    temp = synth_temp_24h(n, seed=seed+1)
    rh_pct, ah_gm3 = synth_humidity_24h(temp, rain, seed=seed+2)

    # ---- Attenuations (P.618 composition) ----
    A_comm = np.array([total_attenuation_db(r, P.comm_freq_GHz, P.elevation_deg,
                                            use_itu_libs=P.use_itu_libs,
                                            add_gas=P.add_gas_p676, add_cloud=P.add_cloud_p840, add_scint=P.add_scintillation_p618,
                                            use_p618=True, use_rss_composition=P.use_p618_rss_composition)
                       for r in rain])
    A_wpt  = np.array([total_attenuation_db(r, P.wpt_freq_GHz,  P.elevation_deg,
                                            use_itu_libs=P.use_itu_libs,
                                            add_gas=P.add_gas_p676, add_cloud=P.add_cloud_p840, add_scint=P.add_scintillation_p618,
                                            use_p618=True, use_rss_composition=P.use_p618_rss_composition)
                       for r in rain])

    # ---- Noise & signal ----
    N_dBm_nom = noise_floor_dBm(P.comm_bw_MHz, P.noise_figure_dB)
    S_dBm = N_dBm_nom + (P.comm_baseline_snr_db - A_comm)

    # ---- Small-scale fading (optional AR(1) dB offset) ----
    if getattr(P, "use_small_fading", False):
        rho  = getattr(P, "fading_ar1_rho", 0.95)
        sig  = getattr(P, "fading_sigma_dB", 3.0)
        fad  = np.zeros_like(S_dBm)
        z    = np.random.default_rng(seed+777).normal(0.0, sig, size=len(S_dBm))
        for i in range(1, len(S_dBm)):
            fad[i] = rho*fad[i-1] + math.sqrt(max(1e-6, 1-rho**2))*z[i]
        S_dBm = S_dBm + fad

    # ---- WPT chain per-minute ----
    Prf_on_dBm = np.zeros(n); Pdc_on_W = np.zeros(n); PL_wpt = np.zeros(n)
    for t in range(n):
        Prf_on_dBm[t], Pdc_on_W[t], PL_wpt[t] = wpt_chain(P, A_wpt[t])
    I_on_dBm = P.wpt_tx_power_dbm - PL_wpt - P.iso_wpt_to_comm_dB - P.rx_filter_rejection_dB + P.nearfield_coupling_dB

    # ---- Base capacity (no-BLER, WPT off) ----
    S_lin_off = 10**(S_dBm/10.0) / (10**(N_dBm_nom/10.0) + 10**(-200.0/10.0))
    se_off = np.log2(1.0 + np.maximum(S_lin_off, 1e-12))
    cap_no_bler_kbps = (P.bearer_bw_kHz/1000.0)*1000.0 * se_off * P.phy_efficiency

    # ---- Two-Stage hourly s_bar ----
    slots_per_hour = int(3600.0 / P.slot_seconds)  # 240 slots per hour for 15-second slots
    s_bar_hour = np.zeros(24)
    for h in range(24):
        sl = slice(h*slots_per_hour, (h+1)*slots_per_hour)
        fav = np.mean(Pdc_on_W[sl]) / max(1e-6, P.load_power_W*0.8)
        fav = np.clip(fav, 0.0, 1.2)
        s_bar_hour[h] = np.clip(fav, P.tws_min_s, P.tws_max_s)

    # ---- Baselines ----
    baselines = list(P.selected_algorithms)
    results = {name: {
        "s": np.zeros(n), "br": np.zeros(n, dtype=int),
        "sinr_db": np.zeros(n), "bler": np.zeros(n), "cap_kbps": np.zeros(n),
        "rho": np.zeros(n), "queue_ms": np.zeros(n), "rtt_ms": np.zeros(n), "mos": np.zeros(n),
        "soc_pct": np.zeros(n),
    } for name in baselines}

    # ---- States ----
    queues = {name: BitQueue() for name in baselines}
    socs   = {name: 60.0 for name in baselines}
    bandit = {"lambda_E": 0.0, "lambda_Q": 0.0, "hour": 0}
    last_action = {name: (0.0, min(P.codec_bitrates_kbps)) for name in baselines}

    # ---- Main loop ----
    for t in range(n):
        hour = t // slots_per_hour
        for name in baselines:
            Q = queues[name]; SOC = socs[name]

            if name == "greedy":
                s, br = choose_greedy_mos(P, Q.queue_kbits, P.base_rtt_ms,
                                          float(S_dBm[t]), float(N_dBm_nom), float(I_on_dBm[t]),
                                          float(cap_no_bler_kbps[t]))

            elif name == "energy_first":
                s, br = choose_energy_first(P, float(SOC), float(cap_no_bler_kbps[t]),
                                            float(S_dBm[t]), float(N_dBm_nom), float(I_on_dBm[t]),
                                            P.base_rtt_ms)

            elif name == "twostage":
                sbar = s_bar_hour[hour]
                s, br = choose_twostage(P, float(sbar),
                                        float(S_dBm[t]), float(N_dBm_nom), float(I_on_dBm[t]),
                                        float(cap_no_bler_kbps[t]), P.base_rtt_ms)

            elif name == "rfwf":
                ls, lbr = last_action[name]
                H = int(getattr(P, "rfwf_horizon", 30))
                s, br = choose_rfwf(P, t, ls, lbr, H,
                                    S_dBm[t:min(n, t+H)], N_dBm_nom,
                                    I_on_dBm[t:min(n, t+H)],
                                    cap_no_bler_kbps[t:min(n, t+H)],
                                    Pdc_on_W[t:min(n, t+H)])

            elif name == "dp2_modeplan":
                sbar = s_bar_hour[hour]
                hstart = hour*60
                hend = min(n, (hour+1)*60)
                s, br = choose_dp2_modeplan(P, t, hour,
                                            S_dBm[hstart:hend], N_dBm_nom,
                                            I_on_dBm[hstart:hend],
                                            cap_no_bler_kbps[hstart:hend],
                                            Pdc_on_W[hstart:hend], float(sbar))

            elif name == "dp2_cvar_rfwf":
                sbar = s_bar_hour[hour]
                hstart = hour*60
                hend = min(n, (hour+1)*60)
                s, br = choose_dp2_cvar_rfwf(P, t, hour,
                                             S_dBm[hstart:hend], N_dBm_nom,
                                             I_on_dBm[hstart:hend],
                                             cap_no_bler_kbps[hstart:hend],
                                             Pdc_on_W[hstart:hend], float(sbar))

            elif name == "sw_idle":
                sbar = s_bar_hour[hour]
                H = int(getattr(P, "sw_idle_horizon", 30))
                s, br = choose_sw_idle(P, t, H,
                                       S_dBm[t:min(n, t+H)], N_dBm_nom,
                                       I_on_dBm[t:min(n, t+H)],
                                       cap_no_bler_kbps[t:min(n, t+H)],
                                       Pdc_on_W[t:min(n, t+H)], float(sbar))

            elif name == "sw_twostage":
                ls, lbr = last_action[name]
                sbar = s_bar_hour[hour]
                H = int(getattr(P, "sw_twostage_horizon", 30))
                s, br = choose_sw_twostage(P, t, ls, lbr, H, float(sbar),
                                        S_dBm[t:min(n, t+H)], N_dBm_nom,
                                        I_on_dBm[t:min(n, t+H)],
                                        cap_no_bler_kbps[t:min(n, t+H)])

            else:
                s, br = choose_random(P, np.random.default_rng(1000+t))

            # epsilon exploration
            s, br = maybe_explore(s, br, P.s_candidates, P.codec_bitrates_kbps,
                                  P.epsilon_explore, np.random.default_rng(10000+t))

            # Link budget with chosen action
            bler_eff = _bler_mixed(float(S_dBm[t]), float(N_dBm_nom), float(I_on_dBm[t]), float(s),
                                   P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
            snr_eff = _snr_eff_capacity_avg(float(S_dBm[t]), float(N_dBm_nom), float(I_on_dBm[t]), float(s))
            SINR_dB = 10.0*np.log10(max(snr_eff, 1e-12))
            cap_kbps = float(cap_no_bler_kbps[t]) * (1.0 - bler_eff)

            arrival_kbps = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            qupd = queues[name].step(arrival_kbps, cap_kbps, slot_seconds=P.slot_seconds)
            rtt_ms = P.base_rtt_ms + qupd["queue_ms"]
            rho = np.clip(arrival_kbps / max(cap_kbps, 1e-6), 0.0, 1.0)
            Ie, Bpl = _impair_from_table(int(br), P.codec_impairments)
            mos = mos_from_emodel(rtt_ms, bler_eff, Ie, Bpl,
                                  jitter_buf_ms=P.jitter_buffer_ms, plc_gain_Bpl=P.plc_gain_Bpl)

            # Energy update
            P_net = float(s)*float(Pdc_on_W[t]) - P.load_power_W
            socs[name] = step_battery_soc(SOC, P_net, P.slot_seconds, P.batt_capacity_mAh,
                                          P.batt_coulomb_eff, P.batt_Rint_ohm, P.batt_voltage_nom_V)

            # Bandit dual update
            if name == "bkb":
                bandit["lambda_E"] = max(0.0, bandit["lambda_E"] + P.bkb_eta_E * bandit.get("last_charge_gap", 0.0))
                bandit["lambda_Q"] = max(0.0, bandit["lambda_Q"] + P.bkb_eta_Q * bandit.get("last_mos_gap", 0.0))
                bandit["hour"] = hour

            # Record
            results[name]["s"][t] = s
            results[name]["br"][t] = br
            results[name]["sinr_db"][t] = SINR_dB
            results[name]["bler"][t] = bler_eff
            results[name]["cap_kbps"][t] = cap_kbps
            results[name]["rho"][t] = rho
            results[name]["queue_ms"][t] = qupd["queue_ms"]
            results[name]["rtt_ms"][t] = rtt_ms
            results[name]["mos"][t] = mos
            results[name]["soc_pct"][t] = socs[name]
            last_action[name] = (float(s), int(br))

    # Save CSV if requested
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name in baselines:
            path = os.path.join(out_dir, f"metrics_{name}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                hdr = ["slot","rain_mmph","temp_C","RH_pct","AH_gm3","A_comm_dB","A_wpt_dB",
                       "S_dBm","I_on_dBm","Prf_on_dBm","Pdc_on_W",
                       "s","bitrate_kbps","SINR_dB","BLER","cap_kbps",
                       "rho","queue_ms","rtt_ms","MOS","SOC_pct"]
                w.writerow(hdr)
                for t in range(n):
                    w.writerow([
                        t, float(rain[t]), float(temp[t]), float(rh_pct[t]), float(ah_gm3[t]),
                        float(A_comm[t]), float(A_wpt[t]), float(S_dBm[t]), float(I_on_dBm[t]),
                        float(Prf_on_dBm[t]), float(Pdc_on_W[t]),
                        float(results[name]["s"][t]), int(results[name]["br"][t]),
                        float(results[name]["sinr_db"][t]), float(results[name]["bler"][t]),
                        float(results[name]["cap_kbps"][t]), float(results[name]["rho"][t]),
                        float(results[name]["queue_ms"][t]), float(results[name]["rtt_ms"][t]),
                        float(results[name]["mos"][t]), float(results[name]["soc_pct"][t]),
                    ])

    # Return sim bundle
    return {
        "env": {"rain": rain, "temp": temp, "rh_pct": rh_pct, "ah_gm3": ah_gm3},
        "A_comm": A_comm, "A_wpt": A_wpt, "S_dBm": S_dBm, "I_on_dBm": I_on_dBm,
        "Prf_on_dBm": Prf_on_dBm, "Pdc_on_W": Pdc_on_W,
        "cap_no_bler_kbps": cap_no_bler_kbps,
        "results": results, "baselines": baselines,
        "out_dir": out_dir, "s_bar_hour": s_bar_hour, "params": P
    }

# Optional demo: keep identical to old main when executed directly
if __name__ == "__main__":
    from plots import plot_time_compare, plot_rain_compare
    P = SimParams()
    sim = run_simulation(P, seed=2025, out_dir="results_day")
    plot_time_compare(sim, save_path=os.path.join("results_day","fig_time_compare.png"))
    plot_rain_compare(sim, save_path=os.path.join("results_day","fig_rain_compare.png"))
    print("Done.")
