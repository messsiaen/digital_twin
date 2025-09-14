# scripts/record_json.py
import os, json, time, argparse, pathlib
import numpy as np
from params import SimParams
from main import run_simulation

"""
输出 schema（大屏按此读取）：
{
  "ts": "2025-09-13T14:33:03Z",
  "meta": { "algo": "twostage", "slot_seconds": 15, "n": 5760 },
  "series": {
    "rain": [[timestamp_ms, value], ...],
    "sinr": [[timestamp_ms, value], ...],
    "duty": [[timestamp_ms, value], ...],
    "mos":  [[timestamp_ms, value], ...]
  }
}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', default='sw_twostage', help='which algorithm result to record')
    ap.add_argument('--out', default='data/metrics.json', help='output json path (relative to repo root)')
    ap.add_argument('--seed', type=int, default=None, help='optional seed')
    ap.add_argument('--start_now', action='store_true',
                    help='use now() as the first timestamp; otherwise start from now - n*step_ms')
    args = ap.parse_args()

    P = SimParams()
    sim = run_simulation(P, seed=args.seed if args.seed is not None else 2025)
    if args.algo not in sim['results']:
        raise SystemExit(f"algo '{args.algo}' not found. Available: {list(sim['results'].keys())}")

    rain    = np.asarray(sim['env']['rain'], dtype=float)
    R       = sim['results'][args.algo]
    sinr_db = np.asarray(R['sinr_db'], dtype=float)
    duty    = np.asarray(R['s'], dtype=float)
    mos     = np.asarray(R['mos'], dtype=float)

    n = int(min(len(rain), len(sinr_db), len(duty), len(mos)))
    step_ms = int(P.slot_seconds * 1000)

    # 时间轴：等间隔（slot_seconds），单位毫秒
    now_ms = int(time.time() * 1000)
    if args.start_now:
        t0 = now_ms
    else:
        t0 = now_ms - (n - 1) * step_ms

    series = {
        'rain': [[t0 + i*step_ms, float(rain[i])]    for i in range(n)],
        'sinr': [[t0 + i*step_ms, float(sinr_db[i])] for i in range(n)],
        'duty': [[t0 + i*step_ms, float(duty[i])]    for i in range(n)],
        'mos':  [[t0 + i*step_ms, float(mos[i])]     for i in range(n)],
    }

    payload = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now_ms/1000)),
        'meta': { 'algo': args.algo, 'slot_seconds': P.slot_seconds, 'n': n },
        'series': series
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding='utf-8')
    print(f"wrote {out_path}  (points={n}, step_ms={step_ms})")

if __name__ == '__main__':
    main()
