from __future__ import annotations
from pathlib import Path
import pandas as pd

from params import SimParams
from modeling import synth_rain_attenuation, step_physics, PesqInterface
from optimizer import choose_action_by_policy, reset_episode

def run_episode_for_policy(P: SimParams, policy: str, rain, vis) -> pd.DataFrame:
    """Run one episode for a given policy"""
    state = {"S": float(P.S0), "T": float(P.T0_C)}
    pesq = PesqInterface(P)
    rows = []
    prev_action = None

    for t in range(len(rain)):
        action = choose_action_by_policy(
            policy, P, t, state, int(vis[t]), float(rain[t])
        )
        u, P_tx_W, tau, subband_idx, beamwidth_idx, pol_tx, waveform_type, ris_phase_opt = action
        res = step_physics(
            P, state, int(vis[t]), float(rain[t]),
            u=u, P_tx_W=P_tx_W, tau=tau,
            subband_idx=subband_idx,
            beamwidth_idx=beamwidth_idx,
            pol_tx=pol_tx,
            waveform_type=waveform_type,
            ris_phase_opt=ris_phase_opt,
            prev_action=prev_action,
            pesq=pesq
        )

        rows.append({"t": t, "policy": policy, **res})
        state["S"] = res["S_next"]
        state["T"] = res["T_next"]
        prev_action = {"P_tx_W": P_tx_W, "tau": tau}

    df = pd.DataFrame(rows)
    return df

def run_episodes(P: SimParams):
    """Run episodes for all policies and save results."""
    n = P.n_slots
    out_dir = Path(P.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate environment
    rain = synth_rain_attenuation(P, n, P.seed)
    vis = P.rand_visibility(n, P.seed + 123)

    dfs = {}
    for pol in P.policies:
        reset_episode(pol)
        df_pol = run_episode_for_policy(P, pol, rain, vis)
        (out_dir / f"sim_timeseries_{pol}.csv").write_text(df_pol.to_csv(index=False))
        dfs[pol] = df_pol
        print(f"[{pol}] Episode complete: {len(df_pol)} slots, "
              f"avg E_harv={df_pol['E_harv_J'].mean():.4f} J, "
              f"avg J={df_pol['J_t'].mean():.4f}")

    return dfs
