# params.py
from dataclasses import dataclass, field
from typing import Tuple, Dict

@dataclass
class SimParams:
    # ---------- Carrier & bandwidth ----------
    comm_freq_GHz: float = 20.0
    comm_bw_MHz: float = 10.0
    noise_figure_dB: float = 7.0
    # Clear-sky baseline SNR (before rain/WPT), adjusted by attenuation each minute
    comm_baseline_snr_db: float = 16.5

    # ---------- Voice bearer (capacity slice) ----------
    bearer_bw_kHz: float = 90.0
    header_overhead_kbps: float = 16.0
    phy_efficiency: float = 0.85

    # ---------- WPT ----------
    wpt_freq_GHz: float = 5.8
    wpt_tx_power_dbm: float = 42.0  # 提高WPT功率，增加充电收益
    wpt_distance_km: float = 0.003
    wpt_pointing_loss_db: float = 2.0

    # Isolation & filtering (WPT->Rx interference) - 降低干扰鼓励充电
    iso_wpt_to_comm_dB: float = 50.0  # 提高隔离度，减少WPT对通信的干扰
    rx_filter_rejection_dB: float = 35.0  # 增强滤波，进一步减少干扰
    nearfield_coupling_dB: float = 3.0  # 降低近场耦合，减少干扰

    # Near/Far-field path model for WPT coupling
    use_wpt_nearfield_model: bool = True
    nf_ref_factor_over_lambda: float = 1.0/(2.0*3.141592653589793)  # r_c = λ/(2π)
    nf_slope_dB_per_dec: float = 60.0
    ff_use_fspl: bool = True

    # ---------- Battery / load ----------
    batt_capacity_mAh: float = 1000.0
    batt_voltage_nom_V: float = 3.7
    batt_Rint_ohm: float = 0.15
    batt_coulomb_eff: float = 0.98
    load_power_W: float = 0.60  # 增加负载，迫使系统更积极充电

    # ---------- Rectifier & DC-DC ---------- 提高效率鼓励充电
    dcdc_eta_min: float = 0.60  # 提高最低效率
    dcdc_eta_max: float = 0.96  # 提高最高效率
    dcdc_eta_pivot_W: float = 0.03  # 降低转折点，小功率时效率更好
    dcdc_eta_steep: float = 15.0  # 更陡峭的效率曲线

    # Soft knee of rectifier efficiency (logistic gate) - 降低阈值鼓励充电
    rect_knee_dbm: float = -30.0  # 降低整流阈值，更容易产生有效充电功率
    rect_knee_width_db: float = 1.5  # 更陡峭的开关特性

    # ---------- ITU switches (attenuation composition) ----------
    use_itu_libs: bool = False
    add_gas_p676: bool = True
    add_cloud_p840: bool = True
    add_scintillation_p618: bool = True
    use_p618_rss_composition: bool = True  # RSS mix of impairments

    # ---------- Fading & phase noise (EVM floor) ----------
    use_small_fading: bool = True
    fading_ar1_rho: float = 0.95
    fading_sigma_dB: float = 3.0
    max_doppler_Hz: float = 50.0
    evm_phase_pct_at_max_doppler: float = 1.5

    # ---------- BLER mapping ----------
    bler_sinr_th_dB: float = 9.5
    bler_slope: float = -1.0
    harq_max_tx: int = 2

    # ---------- Jitter buffer / PLC ----------
    jitter_buffer_ms: float = 40.0
    plc_gain_Bpl: float = 15.0

    # ---------- E-model codec impairments ----------
    codec_impairments: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        6:  (20.0, 10.0),
        12: (15.0, 15.0),
        24: (12.0, 20.0),
        32: (10.0, 25.0),
    })
    codec_bitrates_kbps: Tuple[int, ...] = (6, 12, 24, 32)
    voice_overhead_frac: float = 0.20
    fec_overhead_frac: float = 0.00

    # ---------- SINR effective mixing ----------
    sinr_effective_mode: str = "capacity"   # or "eesm"
    eesm_beta: float = 5.0

    # ---------- Action set & exploration ----------
    s_candidates: Tuple[float, ...] = (0.0, 0.33, 0.66, 1.0)
    epsilon_explore: float = 0.05

    # ---------- Time & Simulation ----------
    slot_seconds: float = 15.0  # Time slot duration in seconds (changed from 60.0 to 15.0)
    simulation_duration_hours: float = 24.0  # Total simulation duration in hours
    total_slots: int = int(24.0 * 3600.0 / 15.0)  # Total number of slots in 24h (5760 slots)
    
    # ---------- Misc ----------
    elevation_deg: float = 10.0
    base_rtt_ms: float = 80.0
    qscale_ms: float = 40.0  # queueing delay scaling

    # ---------- Greedy MOS knobs ---------- 增强充电偏好
    greedy_bitrate_bonus_scale: float = 0.15  # 降低码率奖励
    greedy_charge_bias: float = 0.15  # 大幅增加充电偏好

    # ---------- Safe-Rectifier Opportunistic ---------- 放宽门槛鼓励充电
    sr_mos_floor: float = 3.4  # 降低MOS底线
    sr_bler_max: float = 0.15  # 放宽BLER上限
    sr_rho_max: float = 0.85  # 提高利用率上限
    sr_rect_margin_db: float = 0.5  # 降低整流门槛要求

    # ---------- MPC-lite ----------
    mpc_H: int = 6                 # lookahead in minutes
    mpc_beam_width: int = 3        # beam size
    mpc_switch_cost: float = 0.05  # penalty for (s,br) change
    mpc_lambda_E: float = 0.03     # weight on energy deficit (W*min)

    # ---------- Budgeted-Knapsack Bandit ----------
    # per-hour budgets and dual steps
    bkb_target_charge_Wmin: float = 12.0 * 60.0   # want ~12 W-min energy per hour (example)
    bkb_target_mos: float = 3.7                   # minimum acceptable per-minute MOS (soft)
    bkb_eta_E: float = 2e-4                       # dual step for energy
    bkb_eta_Q: float = 2e-3                       # dual step for MOS
    bkb_switch_cost: float = 0.03                 # small switching penalty

    # ---------- Two-Stage "Charge-Waterfilling" ----------
    # s_bar[h] derived in main from hourly means; these only shape the mapping
    tws_min_s: float = 0.0
    tws_max_s: float = 1.0
    tws_target_frac: float = 0.70   # pick bitrate to ~75% of cap under chosen s

    # ---------- Algorithm Selection ----------
    # Available algorithms in optimizer.py:
    # - "dp2_modeplan": DP2-ModePlan baseline (per-hour block planning)
    # - "rfwf": Rain-Forecast Window Filler
    # - "dp2_cvar_rfwf": DP2 with CVaR-aware placement (risk-averse)

    # - "sw_idle": Sliding-window idle-aware scheduler
    # - "twostage": Two-Stage (hourly s_bar + minute waterfilling)
    # - "sw_twostage": Two-stage with simple dual weights for energy/queue/BLER

    # - "greedy": MOS greedy algorithm
    # - "energy_first": Energy-first SOC hysteresis controller
    # - "seo": Slack-Energy Opportunist (MOS-free)
    # - "bohc": Budget-Only Hourly Charger (MOS-free)
    # - "random": Random baseline
    # Currently selected algorithms to run
    selected_algorithms: Tuple[str, ...] = (
        "sw_twostage"
    )

