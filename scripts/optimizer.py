from __future__ import annotations
from typing import Dict, Tuple
from collections import deque
import numpy as np

from params import SimParams
from modeling import step_physics, PesqInterface

# Internal context for policy algorithms
_CTX = {
    "rain_hist": deque(maxlen=64),
    "dp_plan_masks": {},
    "pesq_ifc": None,
    "prev_action": None,
}

def reset_episode(policy_name: str = "ALL"):
    """Clear internal state at episode start"""
    _CTX["rain_hist"].clear()
    _CTX["dp_plan_masks"].clear()
    _CTX["prev_action"] = None

def _pesq_ifc(P: SimParams) -> PesqInterface:
    if _CTX["pesq_ifc"] is None:
        _CTX["pesq_ifc"] = PesqInterface(P)
    return _CTX["pesq_ifc"]

# Grid search over extended action space
def _grid_search_action(P: SimParams, state: Dict, rain_db_t: float, vis_t: int,
                        tau_levels=None, p_fracs=None) -> Tuple:
    """Grid search to maximize objective over action space"""
    if not vis_t:
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    if tau_levels is None:
        tau_levels = P.gs_tau_levels
    if p_fracs is None:
        p_fracs = P.gs_p_fracs

    best_obj = float('inf')
    best_action = (0, 0.0, 0.0, 0, 0, "V", "CW", False)
    found_feasible = False

    waveform_options = P.waveform_families
    ris_options = [False, True] if P.ris_enable else [False]
    subband_options = [0]
    beamwidth_options = [0]
    pol_options = P.polarizations

    for tau in tau_levels:
        tau = float(min(max(0.0, tau), 1.0, P.latency_tau_max()))
        if tau <= 0:
            continue

        for p_frac in p_fracs:
            P_tx = float(p_frac) * P.P_max_W
            if P_tx <= 1e-12:
                continue

            for subband_idx in subband_options:
                for beamwidth_idx in beamwidth_options:
                    for pol_tx in pol_options:
                        for waveform_type in waveform_options:
                            for ris_opt in ris_options:
                                # Simulate this action
                                res = step_physics(
                                    P, state, vis_t, rain_db_t,
                                    u=1, P_tx_W=P_tx, tau=tau,
                                    subband_idx=subband_idx,
                                    beamwidth_idx=beamwidth_idx,
                                    pol_tx=pol_tx,
                                    waveform_type=waveform_type,
                                    ris_phase_opt=ris_opt,
                                    prev_action=_CTX.get("prev_action"),
                                    pesq=_pesq_ifc(P)
                                )

                                # Check feasibility
                                if not res["feas_all"]:
                                    continue

                                found_feasible = True
                                # Objective value
                                J = res["J_t"]

                                if J < best_obj:
                                    best_obj = J
                                    best_action = (
                                        res["u"], res["P_tx_W"], res["tau"],
                                        subband_idx, beamwidth_idx, pol_tx,
                                        waveform_type, ris_opt
                                    )

    if not found_feasible and vis_t:
        tau_try = 0.5
        P_tx_try = P.P_max_W * 0.5
        res = step_physics(
            P, state, vis_t, rain_db_t,
            u=1, P_tx_W=P_tx_try, tau=tau_try,
            subband_idx=0, beamwidth_idx=0, pol_tx="V",
            waveform_type="Multisine", ris_phase_opt=False,
            prev_action=_CTX.get("prev_action"),
            pesq=_pesq_ifc(P)
        )
        best_action = (res["u"], res["P_tx_W"], res["tau"], 0, 0, "V", "Multisine", False)

    return best_action

# Algorithm 1: Sliding Window (SW)
def choose_action_SW(P: SimParams, t: int, state: Dict, vis_t: int, rain_db_t: float) -> Tuple:
    """Sliding window: charge when predicted rain is low"""
    if not vis_t:
        _CTX["rain_hist"].append(float(rain_db_t))
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    _CTX["rain_hist"].append(float(rain_db_t))
    K = max(1, int(P.sw_window))
    rain_pred = float(np.mean(list(_CTX["rain_hist"])[-K:]))

    # More aggressive rain-adaptive strategy
    # rain_pred: 0-4 dB -> full/high power, 4-6 dB -> moderate, 6+ dB -> reduced
    if rain_pred < 2.0:
        rain_factor = 1.0  # Good conditions - full power
    elif rain_pred < 8.0:
        # Heavy rain: 5dB -> 0.7, 7dB -> 0.4
        rain_factor = 0.6
    else:
        # Very heavy rain: still charge at 40%
        rain_factor = 0.3

    # Check thermal state - if too hot, reduce tau or skip charging
    T_current = float(state.get("T", P.T0_C))
    thermal_headroom = P.T_max_C - T_current

    # Aggressive thermal management - only throttle when really necessary
    if thermal_headroom < 5.0:  # Very hot - reduce significantly
        tau_base = float(min(0.4, P.latency_tau_max()))
        power_scale_base = 0.5
    elif thermal_headroom < 15.0:  # Moderately hot - reduce moderately
        tau_base = float(min(0.7, P.latency_tau_max()))
        power_scale_base = 0.75
    else:
        # Cool: use full parameters
        tau_base = float(min(max(P.sw_tau, 0.0), 1.0, P.latency_tau_max()))
        power_scale_base = 0.9 if P.sw_energy_first else P.sw_p_cap_fraction

    # Apply rain-adaptive scaling
    tau = tau_base * rain_factor
    power_scale = power_scale_base * rain_factor

    # Only skip charging if tau becomes too small
    if tau <= 0.05:  # Less than 5% duty cycle
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    P_tx = P.P_max_W * power_scale

    # ========== Polarization heuristic: match receiver polarization ==========
    # Best strategy: match receiver to minimize mismatch loss
    pol_tx = P.pol_rx_default  # matches receiver for χ = 1.0 (no loss)

    # ========== RIS heuristic: enable based on channel conditions ==========
    # Enable RIS phase optimization when:
    # 1. RIS is available AND
    # 2. Channel is good (low rain) - worth the coherent gain
    # In poor conditions, random phase gives sqrt(N) gain; optimized gives ~N^1.5
    if P.ris_enable:
        if rain_pred < 5.0:
            # Good channel: coherent gain worth the optimization
            ris_opt = True
        else:
            # Poor channel: random phase sufficient (less complex)
            ris_opt = False
    else:
        ris_opt = False

    # Use multisine waveform if available (better efficiency)
    waveform = "Multisine" if "Multisine" in P.waveform_families else "CW"

    return (1, float(P_tx), float(tau), 0, 0, pol_tx, waveform, ris_opt)

# Algorithm 1.5: Adaptive SW (ASW)
def choose_action_ASW(P: SimParams, t: int, state: Dict, vis_t: int, rain_db_t: float) -> Tuple:
    """Adaptive sliding window: constraint-aware power calculation"""
    if not vis_t:
        _CTX["rain_hist"].append(float(rain_db_t))
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    _CTX["rain_hist"].append(float(rain_db_t))
    K = max(1, int(P.sw_window))
    rain_pred = float(np.mean(list(_CTX["rain_hist"])[-K:]))

    # Get current state
    T_current = float(state.get("T", P.T0_C))
    S_current = float(state.get("S", P.S0))

    # ========== Step 1: Calculate constraint-limited maximum powers ==========

    # (C1) PFD constraint: PFD_m = P_tx * sigma_m / r_m^2 <= S_max
    # => P_tx_max_pfd = S_max * r_m^2 / sigma_m
    import math
    beamwidth_deg = P.beam_widths_deg[0]  # use narrowest beam
    G_beam_dB = 10.0 * math.log10(30000.0 / max(1.0, beamwidth_deg**2))
    G_beam = 10.0**(G_beam_dB / 10.0)
    sigma_m = G_beam / (4.0 * math.pi)
    r_m = P.protected_points_r_m[0]
    P_tx_max_pfd = P.S_max_W_per_m2 * r_m**2 / sigma_m

    # (C2) Interference constraint: I_wpt = P_tx * psi_leak * g_r <= I_max
    # => P_tx_max_int = I_max / (psi_leak * g_r)
    psi_leak = P.kappa_leak  # assuming same subband (worst case)
    g_r = 1.0 * G_beam / max(1.0, G_beam**0.5)  # simplified
    I_max_W = 1e-3 * 10.0**(P.I_max_dBm / 10.0)
    P_tx_max_int = I_max_W / (psi_leak * g_r)

    # (C3) Thermal constraint: T_next = a*T_current + b*(tau*P_rx) + c <= T_max
    # => tau * P_rx <= (T_max - a*T_current - c) / b
    # => P_tx_max_thermal depends on tau, so we'll check later
    thermal_headroom = P.T_max_C - T_current
    a, b, c = P.thermal_a, P.thermal_b, P.thermal_c
    thermal_budget = max(0, thermal_headroom - (1-a)*T_current + c)  # simplified

    # Take minimum of PFD and interference constraints
    P_tx_max_constraint = min(P_tx_max_pfd, P_tx_max_int, P.P_max_W)

    # ========== Step 2: Estimate P_rx under current rain conditions ==========
    # Approximate path gain with rain attenuation
    # FSPL at 5m, 28 GHz ≈ 54 dB
    fspl_db = 54.2
    total_loss_db = fspl_db + rain_pred  # simplified: FSPL + rain
    G_tx_dB = 10.0 * math.log10(P.g_tx_base_lin)
    G_rx_dB = 10.0 * math.log10(P.g_rx_base_lin)
    G_composite_dB = G_tx_dB + G_rx_dB - total_loss_db
    G_composite = 10.0**(G_composite_dB / 10.0)

    # ========== Step 3: Rain-adaptive power and tau selection ==========
    # Goal: maximize E_harv = tau * dt * eta * P_rx = tau * dt * eta * (P_tx * G_composite)
    # Subject to: P_tx <= P_tx_max_constraint, thermal constraint on tau

    # Start with maximum feasible power
    P_tx_candidate = P_tx_max_constraint

    # More aggressive rain-adaptive power adjustment
    if rain_pred < 3.0:
        rain_power_factor = 1.0  # Good conditions - full power
    elif rain_pred < 5.0:
        # Moderate rain: still maintain high power
        rain_power_factor = 0.95 - 0.1 * (rain_pred - 3.0)  # 0.95 -> 0.75
    elif rain_pred < 7.0:
        # Heavy rain: reduce but stay aggressive
        rain_power_factor = 0.75 - 0.15 * (rain_pred - 5.0)  # 0.75 -> 0.45
    else:
        # Very heavy rain: still charge at 45%
        rain_power_factor = 0.45

    P_tx = min(P_tx_candidate * rain_power_factor, P.P_max_W)

    # ========== Step 4: Calculate optimal tau based on constraints and conditions ==========
    # Estimate P_rx for thermal calculation
    P_rx_estimate = P_tx * G_composite

    # (C1) Thermal constraint: b * tau * P_rx <= thermal_budget
    if P_rx_estimate > 1e-9 and b > 0:
        tau_max_thermal = thermal_budget / (b * P_rx_estimate)
    else:
        tau_max_thermal = 1.0

    # (C2) Latency constraint
    tau_max_latency = P.latency_tau_max()

    # (C3) Physical hard limit
    tau_hard_max = min(tau_max_thermal, tau_max_latency, 1.0)

    # ========== Calculate target tau based on environment variables ==========
    # Goal: balance energy harvesting vs communication needs

    # Factor 1: Thermal utilization (how much thermal budget to use)
    # thermal_headroom in [0, T_max-T_amb] ~ [0, 55]
    # Use sigmoid-like mapping: more headroom -> higher utilization
    # MORE AGGRESSIVE: shift center point lower and steeper slope
    thermal_utilization = 1.0 / (1.0 + math.exp(-0.2 * (thermal_headroom - 10)))
    # thermal_headroom < 5°C -> ~0.27 (cautious)
    # thermal_headroom = 10°C -> ~0.5 (balanced)
    # thermal_headroom > 20°C -> ~0.88 (very aggressive)
    # Steeper slope (0.2 vs 0.15) and lower center (10 vs 15) for more aggressive charging

    # Factor 2: Channel quality (based on rain attenuation)
    # Better channel -> prefer higher tau for energy harvesting
    # rain_pred in [0, 10] dB
    # Use exponential decay: higher rain -> lower tau preference
    channel_quality = math.exp(-rain_pred / 8.0)
    # rain = 0 dB -> 1.0 (excellent)
    # rain = 4 dB -> 0.61 (moderate)
    # rain = 8 dB -> 0.37 (poor)

    # Factor 3: Energy efficiency consideration (for boosting high-efficiency scenarios)
    # Estimate energy efficiency = E_harv / E_consumed
    dt = 5.0  # slot duration
    eta_approx = 0.7  # approximate rectifier efficiency
    E_harv_estimate = tau_hard_max * dt * eta_approx * P_rx_estimate
    E_consumed_estimate = (1 - tau_hard_max) * P.P_comm_W * dt + tau_hard_max * P.P_base_W * dt
    if E_consumed_estimate > 1e-6:
        energy_efficiency = E_harv_estimate / E_consumed_estimate
    else:
        energy_efficiency = 0.0

    # Combine factors: weighted product of thermal and channel quality
    # tau_target = tau_hard_max * (thermal_util^0.5 * channel_qual^0.5)
    combined_factor = (thermal_utilization**0.5) * (channel_quality**0.5)
    tau_target = tau_hard_max * combined_factor

    # Boost tau_target slightly if efficiency is very high (> 10x)
    if energy_efficiency > 10.0:
        tau_target = min(tau_hard_max, tau_target * 1.2)

    # Apply lower bound: always try to harvest some energy if possible
    # MORE AGGRESSIVE: increase minimum tau from 15% to 25%
    tau_min = 0.25  # minimum 25% duty cycle to justify WPT activation
    tau = float(max(tau_min, min(tau_target, tau_hard_max)))

    # ========== Step 5: Skip charging if conditions are poor ==========
    if tau < 0.05 or P_tx < 0.1:
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    # ========== Polarization heuristic: match receiver ==========
    pol_tx = P.pol_rx_default  # maximize χ = 1.0

    # ========== RIS heuristic: constraint-aware optimization ==========
    # Enable RIS optimization when power budget allows and channel is decent
    # High power + good channel -> coherent gain maximizes E_harv
    if P.ris_enable:
        # RIS worth optimizing if P_tx is high and rain is moderate
        if P_tx > 0.5 * P.P_max_W and rain_pred < 6.0:
            ris_opt = True  # coherent gain ~N^1.5
        elif rain_pred < 3.0:
            # Excellent channel: always optimize
            ris_opt = True
        else:
            # Poor conditions: random phase sufficient
            ris_opt = False
    else:
        ris_opt = False

    # Use best waveform
    waveform = "Multisine" if "Multisine" in P.waveform_families else "CW"

    return (1, float(P_tx), float(tau), 0, 0, pol_tx, waveform, ris_opt)

# Algorithm 2: Binary Planning (DP)
def _ensure_dp_plan_for_segment(P: SimParams, segment_idx: int):
    if segment_idx in _CTX["dp_plan_masks"]:
        return
    H = int(max(1, P.dp_horizon))
    m = int(np.clip(round(P.dp_charge_frac * H), 0, H))
    plan = np.zeros(H, dtype=bool)
    remaining = m
    for L in list(P.dp_blocks):
        while remaining >= L:
            placed = False
            for start in range(0, H - L + 1, L):
                if not np.any(plan[start:start+L]):
                    plan[start:start+L] = True
                    remaining -= L
                    placed = True
                    break
            if not placed:
                break
    _CTX["dp_plan_masks"][segment_idx] = plan

def choose_action_DP(P: SimParams, t: int, state: Dict, vis_t: int, rain_db_t: float) -> Tuple:
    """Binary planning: pre-plan charging slots with thermal awareness"""
    if not vis_t:
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    H = int(max(1, P.dp_horizon))
    segment_idx = t // H
    _ensure_dp_plan_for_segment(P, segment_idx)
    idx_in_seg = t % H
    charge = bool(_CTX["dp_plan_masks"][segment_idx][idx_in_seg])

    # Check thermal state before charging
    T_current = float(state.get("T", P.T0_C))
    thermal_headroom = P.T_max_C - T_current

    # If too hot, skip charging regardless of plan
    if thermal_headroom < 3.0:
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    # Adaptive tau based on thermal headroom
    tau_base = float(min(max(P.dp_tau, 0.0), 1.0, P.latency_tau_max()))

    if thermal_headroom < 10.0:
        # Very hot: reduce tau to 50% (increased from 30%)
        tau = min(0.5, tau_base)
        power_scale = 0.7  # increased from 0.5
    elif thermal_headroom < 20.0:
        # Moderately hot: reduce tau to 75% (increased from 60%)
        tau = min(0.75, tau_base)
        power_scale = 0.85  # increased from 0.7
    else:
        # Cool: use planned tau with high power
        tau = tau_base
        power_scale = 0.95  # increased from 0.8

    if not charge or tau <= 0.0:
        return (0, 0.0, 0.0, 0, 0, "V", "CW", False)

    P_tx = P.P_max_W * power_scale

    # ========== Polarization heuristic: match receiver ==========
    pol_tx = P.pol_rx_default  # minimize mismatch loss

    # ========== RIS heuristic: simple thermal-aware decision ==========
    # DP uses longer charging blocks, so RIS coherent gain more valuable
    # Enable when thermal headroom allows sustained high-power operation
    if P.ris_enable:
        if thermal_headroom > 15.0:
            # Cool enough for sustained coherent charging
            ris_opt = True
        else:
            # Hot: save complexity, use random phase
            ris_opt = False
    else:
        ris_opt = False

    waveform = "Multisine" if "Multisine" in P.waveform_families else "CW"

    return (1, float(P_tx), float(tau), 0, 0, pol_tx, waveform, ris_opt)

# Algorithm 3: Grid Search (GS)
def choose_action_GS(P: SimParams, t: int, state: Dict, vis_t: int, rain_db_t: float) -> Tuple:
    """Grid search: minimize cost (maximize energy - penalties)"""
    return _grid_search_action(P, state, rain_db_t, vis_t)

# Main entry point
def choose_action_by_policy(policy: str, P: SimParams, t: int, state: Dict,
                            vis_t: int, rain_db_t: float) -> Tuple:
    """Choose action based on policy"""
    policy = (policy or "SW").upper()

    if policy == "SW":
        action = choose_action_SW(P, t, state, vis_t, rain_db_t)
    elif policy == "ASW":
        action = choose_action_ASW(P, t, state, vis_t, rain_db_t)
    elif policy == "DP":
        action = choose_action_DP(P, t, state, vis_t, rain_db_t)
    elif policy == "GS":
        action = choose_action_GS(P, t, state, vis_t, rain_db_t)
    else:
        action = choose_action_GS(P, t, state, vis_t, rain_db_t)

    # Store action for slew rate tracking
    _CTX["prev_action"] = {
        "P_tx_W": action[1],
        "tau": action[2],
    }

    return action
