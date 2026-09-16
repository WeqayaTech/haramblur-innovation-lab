import glob, hashlib, os, time
from pathlib import Path

IN, OUT, SHARDS = '/workspace/open-images-v7/images/train', \
                  '/workspace/spotlight/raw/oiv7_train', 32
CLAIMS = Path('/workspace/spotlight/claims')

done = {p.stem for p in Path(OUT).glob('*.txt')}
files = []
for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
    files.extend(glob.glob(os.path.join(IN,'**',ext), recursive=True))

tot=[0]*SHARDS; dn=[0]*SHARDS
for p in files:
    if "negative" in p: continue
    s = int(hashlib.md5(p.encode()).hexdigest(),16) % SHARDS
    tot[s]+=1
    if os.path.splitext(os.path.basename(p))[0] in done: dn[s]+=1

now = time.time()
holder = {}
if CLAIMS.is_dir():
    for c in CLAIMS.glob('shard_*'):
        i = int(c.name.split('_')[1])
        age = now - c.stat().st_mtime
        who = c.read_text().strip().split(':')[0]
        holder[i] = (who, age)

print(f"{'sh':>3} {'done':>7} {'of':>7} {'pct':>6}  {'progress':<22} owner")
print("-"*78)
for i in range(SHARDS):
    pct = 100*dn[i]/max(1,tot[i])
    bar = '#'*int(pct/5)
    if i in holder:
        who, age = holder[i]
        own = f"{who} ({int(age)}s ago)" + ("  STALE!" if age > 300 else "")
    elif dn[i] >= tot[i]:
        own = "COMPLETE"
    elif dn[i]:
        own = "-- partial, unclaimed"
    else:
        own = "-- untouched"
    print(f"{i:>3} {dn[i]:>7,} {tot[i]:>7,} {pct:>5.1f}%  {bar:<22} {own}")

D,T = sum(dn), sum(tot)
print(f"\nTOTAL {D:,} / {T:,}  ({100*D/T:.2f}%)   remaining {T-D:,}")
print(f"claimed now: {sorted(holder)}")
