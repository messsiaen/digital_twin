import os, json, time, base64, argparse, urllib.request, urllib.error, pathlib
import numpy as np

from params import SimParams           # 你已有
from main import run_simulation        # 你已有

# ===== GitHub API helpers =====
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

def _put_file(repo, branch, path, payload_dict, token, message):
    content_b64 = base64.b64encode(json.dumps(payload_dict).encode('utf-8')).decode()
    sha = _get_sha(repo, path, branch, token)
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    data = {'message': message, 'content': content_b64, 'branch': branch}
    if sha: data['sha'] = sha
    return _api(url, method='PUT', data=data, token=token)

# ===== Build one run =====
def build_run(algo: str, seed: int | None = None):
    P = SimParams()
    if seed is None:
        seed = int(time.time()) % 10_000_000
    sim  = run_simulation(P, seed=seed)

    if algo not in sim["results"]:
        raise SystemExit(f"[ERROR] algo '{algo}' not found. Available: {list(sim['results'].keys())}")

    env = sim["env"]
    R   = sim["results"][algo]

    rain    = np.asarray(env["rain"], dtype=float).tolist()
    sinr_db = np.asarray(R["sinr_db"], dtype=float).tolist()
    duty    = np.asarray(R["s"], dtype=float).tolist()
    mos     = np.asarray(R["mos"], dtype=float).tolist()

    n = min(len(rain), len(sinr_db), len(duty), len(mos))
    now = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    run_id = f"run_{now}_{algo}"

    run = {
        "id": run_id,
        "algo": algo,
        "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "seed": seed,
        "step_ms": int(P.slot_seconds * 1000),  # 一步一个 slot；回放速度由前端 PLAY_SPEED 调整
        "n": n,
        "meta": {
            "note": "",
            "slot_seconds": P.slot_seconds
        },
        "rain":    rain[:n],
        "sinr_db": sinr_db[:n],
        "duty":    duty[:n],
        "mos":     mos[:n]
    }
    return run

def update_index(repo, branch, token, run_meta):
    """Append or create runs/index.json"""
    path = 'runs/index.json'
    try:
        url = f'https://api.github.com/repos/{repo}/contents/{path}?ref={branch}'
        obj = _api(url, token=token)  # read existing
        import base64, json
        content = base64.b64decode(obj['content']).decode()
        idx = json.loads(content)
        sha = obj.get('sha')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            idx, sha = {"runs":[]}, None
        else:
            raise
    idx["runs"].append(run_meta)
    # write back
    data_b64 = base64.b64encode(json.dumps(idx).encode('utf-8')).decode()
    url_put = f'https://api.github.com/repos/{repo}/contents/{path}'
    payload = {'message':'runs: update index', 'content':data_b64, 'branch':branch}
    if sha: payload['sha'] = sha
    return _api(url_put, method='PUT', data=payload, token=token)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='<owner>/<repo>')
    ap.add_argument('--branch', default='metrics', help='target branch for run files')
    ap.add_argument('--algo', default='twostage')
    ap.add_argument('--note', default='')
    ap.add_argument('--token_env', default='METRICS_TOKEN')
    args = ap.parse_args()

    token = os.environ.get(args.token_env, '')
    if not token:
        raise SystemExit(f"[ERROR] env {args.token_env} not set")

    run = build_run(args.algo)
    run['meta']['note'] = args.note

    # 1) push the run file
    run_path = f"runs/{run['id']}.json"
    _put_file(args.repo, args.branch, run_path, run, token, message=f"runs: add {run['id']}")

    # 2) update runs/index.json
    run_meta = {"id": run["id"], "algo": run["algo"], "created_at": run["created_at"], "note": run['meta']['note']}
    update_index(args.repo, args.branch, token, run_meta)

    print(f"Published: {run_path} on branch {args.branch}")
    print(f"Index: runs/index.json updated")

if __name__ == '__main__':
    main()
