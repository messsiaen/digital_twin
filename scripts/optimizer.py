# optimizer.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional

# -------------------------------
# Utilities (helpers)
# -------------------------------
def _lin(db):
    return 10.0 ** (db / 10.0)

def _db(lin):
    return -300.0 if lin <= 0 else 10.0 * np.log10(lin)

def _arrival_kbps(br_kbps: int,
                  voice_overhead_frac: float,
                  fec_overhead_frac: float,
                  header_overhead_kbps: float) -> float:
    """Offered rate including overheads."""
    return float(br_kbps) * (1.0 + voice_overhead_frac + fec_overhead_frac) + header_overhead_kbps

def _bler_logistic(sinr_dB: float, th_dB: float, slope: float) -> float:
    """Simple logistic BLER curve vs SINR."""
    x = (sinr_dB - th_dB) * slope
    # sigmoid
    p = 1.0 / (1.0 + np.exp(-x))
    # map to [0,1] with mild flooring to avoid exact 0/1
    p = np.clip(p, 1e-4, 0.9999)
    return float(p)

def _bler_mixed(S_dBm: float, N_dBm: float, I_on_dBm: float, s: float,
                th_dB: float, slope: float, harq_max_tx: int) -> float:
    """
    Duty-cycled effective BLER with simple HARQ residual:
      p_resid = p ** H
    """
    S_lin = _lin(S_dBm); N_lin = _lin(N_dBm); I_on = _lin(I_on_dBm)
    I_eff_lin = s * I_on + (1.0 - s) * _lin(-200.0)  # -200 dBm ~ off
    sinr_on_dB  = _db(S_lin / (N_lin + I_on))
    sinr_eff_dB = _db(S_lin / (N_lin + I_eff_lin))
    # Use the effective SINR for BLER
    p = _bler_logistic(sinr_eff_dB, th_dB, slope)
    p_resid = p ** max(1, int(harq_max_tx))
    return float(np.clip(p_resid, 1e-6, 0.999))

def _snr_eff_capacity_avg(S_dBm: float, N_dBm: float, I_on_dBm: float, s: float) -> float:
    """
    Effective linear SNR under duty-mixed interference.
    """
    S_lin = _lin(S_dBm); N_lin = _lin(N_dBm); I_on = _lin(I_on_dBm)
    I_eff = s * I_on + (1.0 - s) * _lin(-200.0)
    return float(S_lin / (N_lin + I_eff))

def _pick_rate_to_target(P, cap_kbps: float, gamma: float) -> int:
    """
    Choose bitrate so that offered load a(R) ~= gamma * cap.
    If none fits, pick the smallest bitrate.
    """
    target = gamma * max(1e-6, cap_kbps)
    best = None
    best_gap = 1e9
    for R in P.codec_bitrates_kbps:
        a = _arrival_kbps(R, P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
        gap = abs(a - target)
        if gap < best_gap:
            best, best_gap = R, gap
    return int(best if best is not None else P.codec_bitrates_kbps[0])

def _impair_from_table(br_kbps: int, table: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    """Return (Ie, Bpl) for bitrate from params table."""
    if br_kbps in table:
        return table[br_kbps]
    # fallback: nearest by value
    keys = sorted(table.keys())
    nearest = min(keys, key=lambda k: abs(k - br_kbps))
    return table[nearest]

def maybe_explore(s: float, br: int, s_candidates, br_candidates,
                  epsilon: float, rng: np.random.Generator):
    """Epsilon-greedy exploration on top of any policy."""
    if epsilon <= 0.0:
        return s, br
    if rng.random() < epsilon:
        s = float(rng.choice(s_candidates))
    if rng.random() < epsilon:
        br = int(rng.choice(br_candidates))
    return s, br

# -------------------------------
# Heuristics (MOS-free except MOS-greedy)
# -------------------------------

def choose_seo(P,
               queue_kbits: float,
               base_rtt_ms: float,
               S_dBm: float, N_dBm: float, I_on_dBm: float,
               cap0_kbps: float,
               last_s: float = 0.0,
               last_br: Optional[int] = None,
               Pdc_on_W: float = 0.0):
    """
    Slack–Energy Opportunist (MOS-free):
    score = +alpha*max(Ceff - a(R),0) - beta*BLER - zeta*max(rho - rho_cap, 0)
            - gamma*switch_cost + delta*(s*Pdc/Pload)
    """
    alpha = getattr(P, "seo_alpha", 1.0)
    beta  = getattr(P, "seo_beta", 0.4)
    zeta  = getattr(P, "seo_zeta_rho", 0.5)
    rho_cap = getattr(P, "seo_rho_cap", 0.8)
    bler_max = getattr(P, "seo_bler_max", 0.2)
    gamma_sw = getattr(P, "seo_gamma_switch", 0.03)
    delta_en = getattr(P, "seo_delta_energy", 0.2)

    best = (0.0, P.codec_bitrates_kbps[0])
    best_score = -1e9
    for s in P.s_candidates:
        # effective BLER & capacity
        bler = _bler_mixed(S_dBm, N_dBm, I_on_dBm, s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        if bler > bler_max:
            continue
        Ceff = cap0_kbps * (1.0 - bler)
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(br, P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            rho = min(a / max(Ceff, 1e-6), 0.999)
            slack_pos = max(Ceff - a, 0.0)
            switch_pen = gamma_sw if (last_br is not None and (br != last_br or abs(s - last_s) > 1e-6)) else 0.0
            energy_bonus = delta_en * (s * Pdc_on_W / max(P.load_power_W, 1e-6))
            score = alpha * slack_pos - beta * bler - zeta * max(0.0, rho - rho_cap) - switch_pen + energy_bonus
            if score > best_score:
                best_score = score
                best = (float(s), int(br))
    return best

def choose_bohc(P,
                bohc_state: Dict,
                last_s: float, last_br: int,
                S_dBm: float, N_dBm: float, I_on_dBm: float,
                cap0_kbps: float, Pdc_on_W: float):
    """
    Budget-Only Hourly Charger (MOS-free).
    score = -lambda_E*(E_dot_tgt - s*Pdc)_+ - kappa_rho*max(rho-rho_cap,0)
            - kappa_b*max(BLER - pmax,0) - switch_cost + eta*s
    Dual update: lambda_E += eta_E*(E_dot_tgt - s*Pdc)_+
    """
    # State init
    if "lambda_E" not in bohc_state:
        bohc_state["lambda_E"] = 0.0
    if "hour" not in bohc_state:
        bohc_state["hour"] = -1

    # Params
    Wmin_per_hour = getattr(P, "bohc_target_charge_Wmin", 18.0*60.0)  # W*min (i.e., 0.3W*60)
    rho_cap = getattr(P, "bohc_rho_cap", 0.8)
    bler_max = getattr(P, "bohc_bler_max", 0.2)
    kappa_rho = getattr(P, "bohc_kappa_rho", 0.6)
    kappa_b   = getattr(P, "bohc_kappa_bler", 0.6)
    switch_cost = getattr(P, "bohc_switch_cost", 0.03)
    eta_energy_bias = getattr(P, "bohc_energy_bias", 0.05)
    eta_E = getattr(P, "bohc_eta_E", 3e-4)

    # Minute target from hourly target
    E_dot_tgt = Wmin_per_hour / 60.0

    lamE = bohc_state["lambda_E"]

    best = (0.0, P.codec_bitrates_kbps[0])
    best_score = -1e9
    for s in P.s_candidates:
        bler = _bler_mixed(S_dBm, N_dBm, I_on_dBm, s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff = cap0_kbps * (1.0 - bler)
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(br, P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            rho = min(a / max(Ceff, 1e-6), 0.999)
            penE = lamE * max(E_dot_tgt - s*Pdc_on_W, 0.0)
            penR = kappa_rho * max(0.0, rho - rho_cap)
            penB = kappa_b * max(0.0, bler - bler_max)
            sw = switch_cost if (br != last_br or abs(s - last_s) > 1e-6) else 0.0
            score = -penE - penR - penB - sw + eta_energy_bias * s
            if score > best_score:
                best_score, best = score, (float(s), int(br))

    # Dual update (online)
    s_sel = best[0]
    deficit = max(E_dot_tgt - s_sel*Pdc_on_W, 0.0)
    bohc_state["lambda_E"] = max(0.0, lamE + eta_E * deficit)
    return best

def choose_rfwf(P,
                t: int,
                last_s: float, last_br: int,
                H: int,
                S_path: np.ndarray, N_dBm_nom: float,
                I_on_path: np.ndarray,
                cap0_path: np.ndarray,
                Pdc_on_path: np.ndarray):
    """
    Rain-Forecast Window Filler (MOS-free).
    In next H minutes, rank minutes by "favorability" for charging:
    fav = (Pdc/Pload) * 1{ BLER<=pmax and rho<=rho_max at s_max with a target rate }
    Set s = s_max at K best minutes, else s=0 (or s_min). Return the first action.
    """
    H = int(max(1, H))
    s_max = max(P.s_candidates)
    s_min = min(P.s_candidates)
    K = int(getattr(P, "rfwf_K", max(1, H//2)))
    gamma = getattr(P, "rfwf_gamma", 0.75)
    rho_max = getattr(P, "rfwf_rho_max", 0.85)
    bler_max = getattr(P, "rfwf_bler_max", 0.25)

    horizon = min(H, len(S_path))
    fav_scores = np.zeros(horizon)
    ok = np.zeros(horizon, dtype=bool)

    # Evaluate favorability for each future minute
    for h in range(horizon):
        bler = _bler_mixed(S_path[h], N_dBm_nom, I_on_path[h], s_max,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff = cap0_path[h] * (1.0 - bler)
        # choose target bitrate (engineering rule, not MOS)
        R_tgt = _pick_rate_to_target(P, Ceff, gamma)
        a = _arrival_kbps(R_tgt, P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
        rho = min(a / max(Ceff, 1e-6), 0.999)
        ok[h] = (bler <= bler_max) and (rho <= rho_max)
        fav_scores[h] = (Pdc_on_path[h] / max(P.load_power_W, 1e-6)) if ok[h] else 0.0

    # Pick top-K minutes for charging within horizon
    idx = np.argsort(-fav_scores)[:K]
    charge_mask = np.zeros(horizon, dtype=bool)
    charge_mask[idx] = ok[idx]

    # Decide current action (h=0)
    if horizon > 0 and charge_mask[0]:
        s = s_max
    else:
        s = s_min

    # bitrate: waterfill to gamma * Ceff(s)
    bler_now = _bler_mixed(S_path[0], N_dBm_nom, I_on_path[0], s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
    Ceff_now = cap0_path[0] * (1.0 - bler_now)
    br = _pick_rate_to_target(P, Ceff_now, gamma)
    return float(s), int(br)

def choose_twostage(P,
                    sbar: float,
                    S_dBm: float, N_dBm: float, I_on_dBm: float,
                    cap0_kbps: float, base_rtt_ms: float):
    """
    Two-Stage (hourly s_bar + minute waterfilling). MOS-free.
    """
    # choose s nearest to sbar
    s_candidates = np.array(P.s_candidates, dtype=float)
    s = float(s_candidates[np.argmin(np.abs(s_candidates - sbar))])

    # bitrate waterfilling to gamma * Ceff(s)
    gamma = getattr(P, "tws_target_frac", 0.75)
    bler = _bler_mixed(S_dBm, N_dBm, I_on_dBm, s,
                       P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
    Ceff = cap0_kbps * (1.0 - bler)
    br = _pick_rate_to_target(P, Ceff, gamma)
    return s, br

# -------------------------------
# MOS-Greedy (score= MOS) —— 唯一使用 MOS 的策略
# -------------------------------
def choose_greedy_mos(P,
                      queue_kbits: float,
                      base_rtt_ms: float,
                      S_dBm: float, N_dBm: float, I_on_dBm: float,
                      cap0_kbps: float):
    """
    Myopic MOS maximization with small biases (kept for comparison).
    Uses E-model (imported) to score actions.
    """
    try:
        from emodel import mos_from_emodel
    except Exception:
        mos_from_emodel = None

    bitrate_bonus = getattr(P, "greedy_bitrate_bonus_scale", 0.2)
    charge_bias = getattr(P, "greedy_charge_bias", 0.05)
    qscale_ms = getattr(P, "qscale_ms", 60.0)

    best = (min(P.s_candidates), P.codec_bitrates_kbps[0])
    best_score = -1e9

    for s in P.s_candidates:
        bler = _bler_mixed(S_dBm, N_dBm, I_on_dBm, s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff = cap0_kbps * (1.0 - bler)
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(br, P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            rho = min(a / max(Ceff, 1e-6), 0.999)
            # queue delay proxy
            Dq = qscale_ms * (rho / max(1e-6, (1.0 - rho)))
            rtt_ms = base_rtt_ms + Dq
            # MOS estimate
            if mos_from_emodel is not None:
                Ie, Bpl = _impair_from_table(br, P.codec_impairments)
                mos = mos_from_emodel(rtt_ms, bler, Ie, Bpl,
                                      jitter_buf_ms=P.jitter_buffer_ms,
                                      plc_gain_Bpl=P.plc_gain_Bpl)
            else:
                # fallback heuristic if E-model not available
                mos = 4.5 - 1.2 * bler - 0.002 * max(0.0, rtt_ms - 50.0) - 0.8 * rho
                mos = float(np.clip(mos, 1.0, 4.5))
            score = mos + bitrate_bonus * (br / max(P.codec_bitrates_kbps)) * (1.0 - rho) + charge_bias * s
            if score > best_score:
                best_score, best = score, (float(s), int(br))
    return best

# -------------------------------
# Energy-First & Random
# -------------------------------
def choose_energy_first(P,
                        SOC_pct: float,
                        cap0_kbps: float,
                        S_dBm: float, N_dBm: float, I_on_dBm: float,
                        base_rtt_ms: float):
    """
    SOC hysteresis controller (MOS-free).
    """
    L = getattr(P, "ef_soc_low", 35.0)
    H = getattr(P, "ef_soc_high", 90.0)
    s_mid = getattr(P, "ef_s_mid", 0.5)
    rho_cap = getattr(P, "ef_rho_cap", 0.85)
    # Duty
    if SOC_pct < L:
        s = max(P.s_candidates)
    elif SOC_pct > H:
        s = min(P.s_candidates)
    else:
        # nearest to s_mid
        sc = np.array(P.s_candidates, dtype=float)
        s = float(sc[np.argmin(np.abs(sc - s_mid))])
    # Bitrate: choose highest that keeps rho <= rho_cap at this s
    bler = _bler_mixed(S_dBm, N_dBm, I_on_dBm, s,
                       P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
    Ceff = cap0_kbps * (1.0 - bler)
    feasible = []
    for br in P.codec_bitrates_kbps:
        a = _arrival_kbps(br, P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
        rho = min(a / max(Ceff, 1e-6), 0.999)
        if rho <= rho_cap:
            feasible.append(br)
    if len(feasible) == 0:
        br = min(P.codec_bitrates_kbps)
    else:
        br = max(feasible)
    return float(s), int(br)

def choose_random(P, rng: np.random.Generator):
    s = float(rng.choice(P.s_candidates))
    br = int(rng.choice(P.codec_bitrates_kbps))
    return s, br

# -------------------------------
# New Baselines: DP2 / SW-Idle / DP2-CVaR-RFWF / ML-Dual-TwoStage
# -------------------------------

# Internal hour plan cache for DP2-like planners
_dp2_plan_cache = {}

def _dp2_binary_blocks(S_hat: int):
    """Decompose target charge minutes into binary-sized contiguous blocks."""
    sizes = []
    remaining = int(max(0, min(60, S_hat)))
    p = 1
    # build powers of two up to 64
    powers = [32,16,8,4,2,1]
    for L in powers:
        if remaining >= L:
            sizes.append(L)
            remaining -= L
    if remaining > 0:
        sizes.append(remaining)
    return sizes

def _minute_tx_score(P, S_dBm_arr, N_dBm_nom, I_on_arr, cap0_arr, s_tx: float = 0.0):
    """
    Favorability of transmitting at each minute (higher is better).
    Uses capacity under s = s_tx with BLER penalty.
    """
    n = len(S_dBm_arr)
    scores = np.zeros(n)
    for i in range(n):
        bler = _bler_mixed(float(S_dBm_arr[i]), float(N_dBm_nom), float(I_on_arr[i]), float(s_tx),
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        scores[i] = float(cap0_arr[i]) * (1.0 - float(bler))
    return scores

def _minute_charge_score(P, tx_score_arr, Pdc_on_W_arr):
    """
    Favorability for charging (higher means better minute to charge).
    Combine strong WPT power and poor TX condition.
    """
    # Normalize to comparable ranges
    ts = np.asarray(tx_score_arr, dtype=float)
    if ts.max() > 0:
        ts_norm = ts / max(1e-6, np.percentile(ts, 90))
    else:
        ts_norm = ts
    pw = np.asarray(Pdc_on_W_arr, dtype=float)
    if pw.max() > 0:
        pw_norm = pw / max(1e-6, np.percentile(pw, 90))
    else:
        pw_norm = pw
    w_pw = getattr(P, "dp2_w_power", 0.65)
    w_bad = getattr(P, "dp2_w_badtx", 0.35)
    # charging is good when tx_score is low -> (1 - ts_norm)
    return w_pw * pw_norm + w_bad * (1.0 - ts_norm)

def _place_blocks_nonoverlap(score_arr, block_sizes):
    """
    Greedy placement: for each block size, place on the contiguous window with maximum score sum,
    without overlap. Returns a boolean mask of 'charge' minutes.
    """
    L = len(score_arr)
    used = np.zeros(L, dtype=bool)
    charge = np.zeros(L, dtype=bool)
    # precompute prefix sums for fast window sums
    ps = np.zeros(L+1)
    ps[1:] = np.cumsum(score_arr)
    for blen in block_sizes:
        best_s, best_val = None, -1e18
        # sliding window
        window_sum = ps[blen:] - ps[:-blen]
        # penalize windows that overlap used minutes
        for s in range(0, L - blen + 1):
            if used[s:s+blen].any():
                continue
            val = float(window_sum[s])
            # slight tie-breaker toward fewer switch fragments (prefer to place adjacent to existing charge)
            l_adj = (s > 0 and charge[s-1])
            r_adj = (s+blen < L and charge[s+blen])
            if l_adj or r_adj:
                val *= 1.02
            if val > best_val:
                best_val, best_s = val, s
        if best_s is not None:
            charge[best_s:best_s+blen] = True
            used[best_s:best_s+blen] = True
    return charge

def _dp2_plan_for_hour(P, hour_idx: int, S_hour: np.ndarray, N_dBm_nom: float, I_hour: np.ndarray,
                       cap0_hour: np.ndarray, Pdc_hour: np.ndarray, sbar: float):
    """
    Build (and cache) a DP2-style charge/tx plan for this hour.
    """
    key = (hour_idx, int(1000*float(sbar)))
    if key in _dp2_plan_cache:
        return _dp2_plan_cache[key]

    S_hat = int(np.clip(round(60.0 * float(sbar)), 0, 60))
    block_sizes = _dp2_binary_blocks(S_hat)

    # Rank minutes by 'charge favorability'
    tx_score = _minute_tx_score(P, S_hour, N_dBm_nom, I_hour, cap0_hour, s_tx=0.0)
    charge_score = _minute_charge_score(P, tx_score, Pdc_hour)

    charge_mask = _place_blocks_nonoverlap(charge_score, block_sizes)
    _dp2_plan_cache[key] = charge_mask
    return charge_mask

def choose_dp2_modeplan(P, t: int, hour: int,
                        S_dBm_path: np.ndarray, N_dBm_nom: float,
                        I_on_path: np.ndarray, cap0_path: np.ndarray,
                        Pdc_on_path: np.ndarray, sbar_hour_val: float):
    """
    DP2-ModePlan baseline: per-hour block planning to reduce switches and match charge share.
    - Uses sbar_hour_val to set target charge minutes.
    - Places blocks on minutes best for charging (high WPT power, poor TX).
    """

    # Plan for current hour
    H = min(60, len(S_dBm_path))
    charge_mask = _dp2_plan_for_hour(P, hour, S_dBm_path[:H], N_dBm_nom, I_on_path[:H],
                                     cap0_path[:H], Pdc_on_path[:H], sbar_hour_val)
    idx = t % 60
    is_charge = bool(charge_mask[idx])
    if is_charge:
        s = float(max(P.s_candidates))
        # Keep bitrate conservative while charging
        # choose highest feasible under rho_cap
        bler = _bler_mixed(float(S_dBm_path[idx]), float(N_dBm_nom), float(I_on_path[idx]), s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff = float(cap0_path[idx]) * (1.0 - bler)
        rho_cap = getattr(P, "dp2_rho_cap", 0.85)
        feasible = []
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            if a <= rho_cap * max(Ceff, 1e-6):
                feasible.append(int(br))
        br = int(min(P.codec_bitrates_kbps) if len(feasible)==0 else max(feasible))
        return float(s), int(br)
    else:
        s = float(min(P.s_candidates))
        gamma = getattr(P, "dp2_gamma_tx", 0.8)
        br = _pick_rate_to_target(P, float(cap0_path[idx]), gamma)
        return float(s), int(br)

def _cvar_tail_mean(x: np.ndarray, q: float):
    """CVaR (tail mean) with tail fraction q in [0,1]."""
    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0 or len(x)==0:
        return float(np.mean(x) if len(x)>0 else 0.0)
    k = max(1, int(np.ceil(q * len(x))))
    xs = np.sort(np.asarray(x).reshape(-1))
    return float(np.mean(xs[-k:]))

def choose_dp2_cvar_rfwf(P, t: int, hour: int,
                         S_dBm_path: np.ndarray, N_dBm_nom: float,
                         I_on_path: np.ndarray, cap0_path: np.ndarray,
                         Pdc_on_path: np.ndarray, sbar_hour_val: float):
    """
    DP2 with CVaR-aware placement (risk-averse): prioritize placing charge blocks on windows
    whose 'badness' for TX has highest tail risk; otherwise acts like DP2-ModePlan.
    """

    H = min(60, len(S_dBm_path))
    tx_score = _minute_tx_score(P, S_dBm_path[:H], N_dBm_nom, I_on_path[:H], cap0_path[:H], s_tx=0.0)
    # 'Badness' can be (1 - normalized tx_score)
    x = tx_score
    x_norm = x / max(1e-6, np.percentile(x, 90))
    bad = 1.0 - x_norm
    # Combine with WPT favorability to encourage charging when power is high
    pw = np.asarray(Pdc_on_path[:H], dtype=float)
    if pw.max() > 0:
        pw_norm = pw / max(1e-6, np.percentile(pw, 90))
    else:
        pw_norm = pw
    risk_q = getattr(P, "dp2_cvar_q", 0.2)

    # Build a per-window score function for block placement
    # We'll compute later inside placement function; here we precompute a 'badness' array
    block_sizes = _dp2_binary_blocks(int(np.clip(round(60.0 * float(sbar_hour_val)), 0, 60)))

    # Custom placement: evaluate each feasible start by CVaR of badness within block plus avg pw
    def place_cvar(bad_arr, pw_arr, sizes):
        L = len(bad_arr)
        used = np.zeros(L, dtype=bool)
        charge = np.zeros(L, dtype=bool)
        ps_bad = np.zeros(L+1); ps_bad[1:] = np.cumsum(bad_arr)  # fallback if CVaR ties
        for blen in sizes:
            best_s, best_val = None, -1e18
            for s0 in range(0, L-blen+1):
                if used[s0:s0+blen].any():
                    continue
                window_bad = bad_arr[s0:s0+blen]
                window_pw  = pw_arr[s0:s0+blen]
                val = 1.0 * _cvar_tail_mean(window_bad, risk_q) + 0.5 * float(np.mean(window_pw))
                # small contiguity bonus
                if (s0>0 and charge[s0-1]) or (s0+blen<L and charge[s0+blen]):
                    val *= 1.02
                if val > best_val:
                    best_val, best_s = val, s0
            if best_s is not None:
                charge[best_s:best_s+blen] = True
                used[best_s:best_s+blen] = True
        return charge

    charge_mask = place_cvar(bad, pw_norm, block_sizes)
    idx = t % 60
    is_charge = bool(charge_mask[idx])
    if is_charge:
        s = float(max(P.s_candidates))
        # conservative bitrate during charging
        bler = _bler_mixed(float(S_dBm_path[idx]), float(N_dBm_nom), float(I_on_path[idx]), s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff = float(cap0_path[idx]) * (1.0 - bler)
        rho_cap = getattr(P, "dp2_rho_cap", 0.85)
        feasible = []
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            if a <= rho_cap * max(Ceff, 1e-6):
                feasible.append(int(br))
        br = int(min(P.codec_bitrates_kbps) if len(feasible)==0 else max(feasible))
        return float(s), int(br)
    else:
        s = float(min(P.s_candidates))
        gamma = getattr(P, "rfwf_gamma", 0.75)
        br = _pick_rate_to_target(P, float(cap0_path[idx]), gamma)
        return float(s), int(br)

def choose_sw_idle(P, t: int, H: int,
                   S_path: np.ndarray, N_dBm_nom: float,
                   I_on_path: np.ndarray, cap0_path: np.ndarray,
                   Pdc_on_path: np.ndarray, sbar_hour_val: float):
    """
    Sliding-window idle-aware scheduler with modified parameters:
    - In next H minutes, select top M = round((1 - sbar_hour_val)*H) minutes by tx favorability to transmit.
    - Others charge with s = s_max.
    - Modified to be more conservative than dp2_modeplan
    """
    H = int(max(1, min(H, len(S_path))))
    tx_score = _minute_tx_score(P, S_path[:H], N_dBm_nom, I_on_path[:H], cap0_path[:H], s_tx=0.0)
    
    # 修改：使用更保守的传输比例，减少传输时隙
    conservative_sbar = min(0.95, sbar_hour_val + 0.15)  # 增加充电比例
    M = int(np.clip(round((1.0 - float(conservative_sbar)) * H), 0, H))
    
    order = np.argsort(-tx_score)  # descending
    tx_mask = np.zeros(H, dtype=bool)
    tx_mask[order[:M]] = True
    idx = 0  # current minute in window
    # within the window, current minute corresponds to index 0
    if tx_mask[idx]:
        s = float(min(P.s_candidates))
        gamma = getattr(P, "sw_idle_gamma", 0.5)  # 更保守的传输容量目标
        br = _pick_rate_to_target(P, float(cap0_path[0]), gamma)
        return float(s), int(br)
    else:
        s = float(max(P.s_candidates))
        # conservative bitrate while charging
        bler = _bler_mixed(float(S_path[0]), float(N_dBm_nom), float(I_on_path[0]), s,
                           P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff = float(cap0_path[0]) * (1.0 - bler)
        rho_cap = getattr(P, "sw_idle_rho_cap", 0.7)  # 更保守的充电容量限制
        feasible = []
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            if a <= rho_cap * max(Ceff, 1e-6):
                feasible.append(int(br))
        br = int(min(P.codec_bitrates_kbps) if len(feasible)==0 else max(feasible))
        return float(s), int(br)

# Lightweight dual-weighted two-stage
_dual_state = {"lambda_E": 0.0, "lambda_Q": 0.0, "lambda_B": 0.0}

def choose_sw_twostage(P,
                     t: int,
                     last_s: float, last_br: int,
                     H: int,
                     sbar_hour_val: float,
                     S_path: np.ndarray, N_dBm_nom: float,
                     I_on_path: np.ndarray,
                     cap0_path: np.ndarray):
    """Two-Stage + SW-Idle strict hybrid (rolling window hard quota).
    In next H slots, select M = round((1 - sbar)*H) transmission slots using a greedy
    adjacency-aware selector; charge in the rest. Then return the decision for the first slot.

    - Transmission candidate uses s_tx = min(P.s_candidates) (low interference).
    - Charging candidate uses s_ch = max(P.s_candidates).
    - Within a chosen mode, pick bitrate to maximize MOS (subject to rho stability).
    - Scores include MOS, BLER soft penalty, and utilization over-cap penalty (idle awareness).
    - Adjacency bonus encourages contiguous selections, reducing switch churn.

    Interface mirrors choose_rfwf: no change to top-level simulation API.
    """
    try:
        from emodel import mos_from_emodel
    except Exception:
        mos_from_emodel = None

    # Hyperparameters (can be overridden via P.*)
    rho_cap    = getattr(P, "swidle_rho_cap", 0.85)
    w_mos      = getattr(P, "swidle_w_mos", 1.0)
    w_bler     = getattr(P, "swidle_w_bler", 0.5)
    w_u_over   = getattr(P, "swidle_w_util_over", 0.8)
    adj_bonus  = getattr(P, "swidle_adj_bonus", 0.05)   # relative bonus factor (scaled by score std)
    qscale_ms  = getattr(P, "qdelay_scale_ms", 30.0)

    s_tx = float(min(P.s_candidates))
    s_ch = float(max(P.s_candidates))

    H = int(max(1, min(H, len(S_path), len(I_on_path), len(cap0_path))))
    M = int(np.clip(round((1.0 - float(sbar_hour_val)) * H), 0, H))

    # Pre-compute per-slot TX scores using best bitrate at s_tx
    base_scores = np.zeros(H, dtype=float)
    tx_choice_br = np.zeros(H, dtype=int)
    tx_choice_mos = np.zeros(H, dtype=float)

    for i in range(H):
        S = float(S_path[i]); I = float(I_on_path[i]); cap0 = float(cap0_path[i])
        bler_tx = _bler_mixed(S, float(N_dBm_nom), I, s_tx, P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff_tx = cap0 * (1.0 - bler_tx)

        best_mos_i = 0.0
        best_br_i = min(P.codec_bitrates_kbps)
        best_util_i = 1e9
        for br in P.codec_bitrates_kbps:
            a = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            util = a / max(Ceff_tx, 1e-6)
            rho_clipped = min(util, 0.999)
            Dq = qscale_ms * (rho_clipped / max(1e-6, (1.0 - rho_clipped)))
            rtt_ms = P.base_rtt_ms + Dq
            if mos_from_emodel is not None:
                Ie, Bpl = _impair_from_table(int(br), P.codec_impairments)
                mos = mos_from_emodel(rtt_ms, bler_tx, Ie, Bpl,
                                      jitter_buf_ms=P.jitter_buffer_ms,
                                      plc_gain_Bpl=P.plc_gain_Bpl)
            else:
                mos = 4.5 - 1.2 * bler_tx - 0.002 * max(0.0, rtt_ms - 50.0) - 0.8 * rho_clipped
                mos = float(np.clip(mos, 1.0, 4.5))
            # choose bitrate that maximizes MOS (tie-break by lower util)
            if (mos > best_mos_i) or (abs(mos - best_mos_i) < 1e-6 and util < best_util_i):
                best_mos_i, best_br_i, best_util_i = mos, int(br), float(util)

        tx_choice_br[i] = best_br_i
        tx_choice_mos[i] = best_mos_i
        u_over = max(0.0, best_util_i - rho_cap)
        base_scores[i] = w_mos * best_mos_i - w_bler * float(bler_tx) - w_u_over * u_over

    # Greedy adjacency-aware selection of M TX slots
    # Normalize adjacency bonus scale by score std to be unitless
    std = float(np.std(base_scores)) if H > 1 else 1.0
    bonus = adj_bonus * (std if std > 1e-6 else 1.0)
    gains = base_scores.copy()
    selected = np.zeros(H, dtype=bool)
    for _ in range(M):
        j = int(np.argmax(gains))
        if selected[j]:
            # If already selected due to equal gains, pick next best unselected
            # (simple guard; worst-case O(H^2) which is fine for small H)
            idxs = np.argsort(-gains)
            j = int([k for k in idxs if not selected[int(k)]][0]) if np.any(~selected) else j
        selected[j] = True
        # spread adjacency bonus to neighbors to promote contiguity
        if j-1 >= 0 and not selected[j-1]:
            gains[j-1] += bonus
        if j+1 < H and not selected[j+1]:
            gains[j+1] += bonus
        gains[j] = -1e18  # do not re-select

    # Decide action for first slot in window
    if selected[0]:
        # Transmission minute: fix s to s_tx, pick bitrate to maximize MOS
        cand = []
        for br in P.codec_bitrates_kbps:
            # reuse precomputed MOS if same br; else recompute quickly
            if int(br) == int(tx_choice_br[0]):
                mos = tx_choice_mos[0]
            else:
                S0 = float(S_path[0]); I0 = float(I_on_path[0]); cap0 = float(cap0_path[0])
                bler0 = _bler_mixed(S0, float(N_dBm_nom), I0, s_tx, P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
                Ceff0 = cap0 * (1.0 - bler0)
                a0 = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
                util0 = a0 / max(Ceff0, 1e-6)
                rho0 = min(util0, 0.999)
                Dq0 = qscale_ms * (rho0 / max(1e-6, (1.0 - rho0)))
                rtt0 = P.base_rtt_ms + Dq0
                if mos_from_emodel is not None:
                    Ie, Bpl = _impair_from_table(int(br), P.codec_impairments)
                    mos = mos_from_emodel(rtt0, bler0, Ie, Bpl,
                                          jitter_buf_ms=P.jitter_buffer_ms,
                                          plc_gain_Bpl=P.plc_gain_Bpl)
                else:
                    mos = 4.5 - 1.2 * bler0 - 0.002 * max(0.0, rtt0 - 50.0) - 0.8 * rho0
                    mos = float(np.clip(mos, 1.0, 4.5))
            cand.append((mos, int(br)))
        cand.sort(key=lambda x: x[0], reverse=True)
        return float(s_tx), int(cand[0][1])
    else:
        # Charging minute: fix s to s_ch, choose the highest feasible bitrate under rho_cap (else min)
        S0 = float(S_path[0]); I0 = float(I_on_path[0]); cap0 = float(cap0_path[0])
        bler0 = _bler_mixed(S0, float(N_dBm_nom), I0, s_ch, P.bler_sinr_th_dB, P.bler_slope, P.harq_max_tx)
        Ceff0 = cap0 * (1.0 - bler0)
        feasible = []
        for br in P.codec_bitrates_kbps:
            a0 = _arrival_kbps(int(br), P.voice_overhead_frac, P.fec_overhead_frac, P.header_overhead_kbps)
            util0 = a0 / max(Ceff0, 1e-6)
            if util0 <= rho_cap:
                feasible.append(int(br))
        if len(feasible) == 0:
            br0 = int(min(P.codec_bitrates_kbps))
        else:
            br0 = int(max(feasible))
        return float(s_ch), int(br0)
