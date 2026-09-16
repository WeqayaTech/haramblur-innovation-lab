import json, glob
from pathlib import Path

RAW  = Path('/workspace/spotlight/raw/oiv7_train')
RUN  = Path('/workspace/spotlight/run/oiv7_train')
TOTAL_IMAGES  = 475_215
DETS_PER_IMG  = 3.324          # measured from a 3,000-file sample
RATE_PER_1K   = 0.321         # measured from collected jobs

# ---- stage 1: labeling -----------------------------------------------
labeled = sum(1 for _ in RAW.glob('*.json'))
dets_made = 1_579_649  # exact, counted from all sidecars 2026-07-29
dets_total = int(TOTAL_IMAGES * DETS_PER_IMG)

# ---- stage 2: verification -------------------------------------------
verified = set()
for f in RUN.glob('verdicts*.jsonl'):
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            verified.add((r['image_stem'], r['det_index']))

inflight = set()
for pf in (RUN / 'batch').glob('pending_*.json'):
    try:
        keys = json.loads(pf.read_text())
    except Exception:
        print(f'  !! truncated reservation {pf.name}')
        continue
    for k in keys:
        inflight.add(tuple(k))
inflight -= verified                       # collected jobs leave their file behind

awaiting = max(0, dets_made - len(verified) - len(inflight))

# ---- money -----------------------------------------------------------
spent = 0.0
jf = RUN / 'batch' / 'jobs.json'
if jf.exists():
    for j in json.loads(jf.read_text()):
        spent += j.get('actual_cost_usd') or (j.get('est_cost_usd', 0)
                                              if not j.get('collected') else 0)

bar = lambda p: '#' * int(p / 5)
print(f"LABELING     {labeled:>9,} / {TOTAL_IMAGES:,} images  "
      f"{100*labeled/TOTAL_IMAGES:5.1f}%  {bar(100*labeled/TOTAL_IMAGES)}")
print(f"  detections {dets_made:>9,} produced   (~{dets_total:,} at completion)")
print()
print(f"VERIFIED     {len(verified):>9,}  {100*len(verified)/max(1,dets_total):5.1f}% of final")
print(f"IN FLIGHT    {len(inflight):>9,}  submitted, awaiting results")
print(f"AWAITING     {awaiting:>9,}  labeled but NOT yet submitted  <-- feed this")
print()
print(f"spent  ~${spent:,.2f}      projected total ~${dets_total*RATE_PER_1K/1000:,.0f}")
print()
if awaiting >= 10_000:
    print(f"=> submit now: {awaiting//10_000} more job(s) of 10k available")
elif awaiting > 0:
    print(f"=> only {awaiting:,} awaiting; wait for labeling to produce more")
else:
    print("=> nothing to submit right now")
