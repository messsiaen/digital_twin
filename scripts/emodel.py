# emodel.py
import numpy as np
from typing import Tuple

def jitter_buffer_transform(rtt_ms: float, plr: float, buf_ms: float) -> Tuple[float,float]:
    """
    Map RTT/PLR to effective one-way delay & late-loss due to buffer.
    """
    one_way = max(0.0, 0.5*rtt_ms)
    late_loss = 0.0 if one_way <= buf_ms else min(1.0, (one_way - buf_ms)/one_way)*0.1
    plr_eff = np.clip(plr + late_loss, 0.0, 1.0)
    return one_way, plr_eff

def mos_from_emodel(rtt_ms: float, plr_frac: float, Ie: float, Bpl: float,
                    Ro: float = 94.2, Is: float = 0.0, A: float = 0.0,
                    jitter_buf_ms: float = 60.0, plc_gain_Bpl: float = 0.0) -> float:
    d, Ppl_eff = jitter_buffer_transform(rtt_ms, plr_frac, jitter_buf_ms)
    Id = 0.024*d + 0.11 * max(0.0, d - 177.3)
    Bpl_eff = max(0.0, Bpl + plc_gain_Bpl)
    Ppl = 100.0 * Ppl_eff
    Ie_eff = Ie + (95.0 - Ie) * (Ppl / (Ppl + Bpl_eff + 1e-9))
    R = Ro - Is - Id - Ie_eff + A
    if R <= 0: return 1.0
    if R >= 100: return 4.5
    return float(1.0 + 0.035*R + 7e-6*R*(R-60)*(100-R))
