# scripts/live_feeder.py
import os, time, json, base64, argparse, pathlib, urllib.request, urllib.error
import numpy as np

from params import SimParams
from main import run_simulation  # 直接用你已有的仿真入口

def _api(url, method='GET', data=None, token=None):
    req = urllib.request.Request(url, method=method)
    req.add_header('Accept', 'application/vnd.github+json')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    if data is not None:
        body = json.dumps(data).encode('utf-8')
        req.add_header('Content-Type', 'application/json')
        req.data = body
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def _get_sha(repo, path, branch, token):
    url = f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}'
    try:
        obj = _api(url, token=token)
        return obj.get('sha')
    except urllib.error.HTTPError as e:
        if e.code == 404: return None
        raise

def push_json_to_gh(repo, branch, path, payload, token, message='update live metrics'):
    content_b64 = base64.b64encode(json.dumps(payload).encode('utf-8')).decode()
    sha = _get_sha(repo, path, branch, token)
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    data = {'message': message, 'content': content_b64, 'branch': branch}
    if sha: data['sha'] = sha
    return _api(url, method='PUT', data=data, token=token)

def lerp(a, b, w): return (1.0-w)*a + w*b
def interp(arr, x):
    i = int(np.floor(x)); w = x - i; j = min(i+1, len(arr)-1)
    return float(lerp(arr[i], arr[j], w))

def build_payload(now_ms, rain_win, sinr_win, duty_win, mos_win):
    ts_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now_ms/1000))
    return {
        "ts": ts_iso,
        "series": {
            "rain": [[t, v] for t, v in rain_win],
            "sinr": [[t, v] for t, v in sinr_win],
            "duty": [[t, v] for t, v in duty_win],
            "mos":  [[t, v] for t, v in mos_win]
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<owner>/<repo> 比如 yourname/your-icassp-repo")
    ap.add_argument("--branch", default="gh-pages")
    ap.add_argument("--gh_path", default="data/metrics.json")
    ap.add_argument("--algo", default="twostage", help="展示哪个算法（需存在于 sim['results']）")
    ap.add_argument("--refresh", type=float, default=5.0, help="刷新周期（秒）")
    ap.add_argument("--slot_interp", action="store_true", help="在slot间做线性插值")
    ap.add_argument("--token_env", default="METRICS_TOKEN", help="PAT放在这个环境变量里")
    ap.add_argument("--window_sec", type=int, default=600, help="展示最近多少秒的曲线（默认10分钟）")
    args = ap.parse_args()

    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"[ERROR] 环境变量 {args.token_env} 未设置（需要 GitHub PAT）")

    # 预生成一段轨迹（24h）
    P = SimParams()
    seed = int(time.time()) % 10_000_000
    sim  = run_simulation(P, seed=seed)
    rain = np.asarray(sim["env"]["rain"])

    if args.algo not in sim["results"]:
        raise SystemExit(f"[ERROR] algo '{args.algo}' 不在结果里，可用: {list(sim['results'].keys())}")

    R = sim["results"][args.algo]
    duty_arr = np.asarray(R["s"])
    sinr_arr = np.asarray(R["sinr_db"])
    mos_arr  = np.asarray(R["mos"])

    n_slots  = len(rain)
    slot_len = float(P.slot_seconds)  # 每个slot多少秒

    # 滑动窗口（只保留最近 window_sec 秒）
    def purge(win, now_ms):
        limit = now_ms - args.window_sec * 1000
        while win and win[0][0] < limit: win.pop(0)

    # “秒”索引（在整段轨迹中的位置）
    sec_idx = 0.0

    while True:
        # 到尾了就换个seed再来一段
        if sec_idx >= n_slots * slot_len:
            seed += 1
            sim  = run_simulation(P, seed=seed)
            rain = np.asarray(sim["env"]["rain"])
            R = sim["results"][args.algo]
            duty_arr = np.asarray(R["s"])
            sinr_arr = np.asarray(R["sinr_db"])
            mos_arr  = np.asarray(R["mos"])
            n_slots  = len(rain)
            sec_idx  = 0.0

        # 取当前秒对应的值（可选插值）
        if args.slot_interp:
            x = sec_idx / slot_len
            x = max(0.0, min(x, n_slots - 1.000001))
            rain_v = interp(rain,     x)
            sinr_v = interp(sinr_arr, x)
            duty_v = interp(duty_arr, x)
            mos_v  = interp(mos_arr,  x)
        else:
            i = int(min(sec_idx // slot_len, n_slots - 1))
            rain_v = float(rain[i]); sinr_v = float(sinr_arr[i])
            duty_v = float(duty_arr[i]); mos_v = float(mos_arr[i])

        now_ms = int(time.time() * 1000)

        # 维护四条时间序列窗口
        if 'wins' not in globals():
            globals()['wins'] = {
                'rain': [], 'sinr': [], 'duty': [], 'mos': []
            }
        wins['rain'].append((now_ms, float(rain_v)))
        wins['sinr'].append((now_ms, float(sinr_v)))
        wins['duty'].append((now_ms, float(duty_v)))
        wins['mos'].append ((now_ms, float(mos_v)))
        purge(wins['rain'], now_ms); purge(wins['sinr'], now_ms)
        purge(wins['duty'], now_ms); purge(wins['mos'],  now_ms)

        payload = build_payload(now_ms, wins['rain'], wins['sinr'], wins['duty'], wins['mos'])

        # 写到 gh-pages 分支的 data/metrics.json
        push_json_to_gh(args.repo, args.branch, args.gh_path, payload, token,
                        message="dashboard: live tick")

        # 步进到下一个5秒节拍
        sec_idx += args.refresh
        time.sleep(args.refresh)

if __name__ == '__main__':
    main()
