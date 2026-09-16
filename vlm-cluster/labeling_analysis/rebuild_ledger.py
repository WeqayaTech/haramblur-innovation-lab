import json
from pathlib import Path
from google import genai
BD  = Path('/workspace/spotlight/run/oiv7_train/batch')
RUN = Path('/workspace/spotlight/run/oiv7_train')
have = set()
for f in RUN.glob('verdicts*.jsonl'):
    for line in open(f):
        r = json.loads(line); have.add((r['image_stem'], r['det_index']))
client = genai.Client(); jobs = []
for b in client.batches.list():
    dn = getattr(b, 'display_name', '') or ''
    if not dn.startswith('spotlight_'): continue
    short = dn[len('spotlight_'):]
    keys = []
    mf = BD / f'meta_{short}.jsonl'
    if mf.exists():
        for line in mf.read_text().splitlines():
            try:
                r = json.loads(line); keys.append((r['image_stem'], r['det_index']))
            except Exception: pass
    jobs.append({"short_id": short, "job_name": b.name,
                 "model": "gemini-3.5-flash-lite", "n_requests": len(keys),
                 "est_cost_usd": round(len(keys)*0.000321, 4), "submitted_at": short,
                 "state": str(b.state),
                 "collected": (not keys) or all(k in have for k in keys),
                 "prompt_version": "rebuilt"})
jobs.sort(key=lambda j: j['short_id'])
(BD/'jobs.json').write_text(json.dumps(jobs, indent=2))
open_j = [j['short_id'] for j in jobs if not j['collected']]
print(f"[ledger] {len(jobs)} jobs on Google, {len(open_j)} awaiting collection: {open_j}")
