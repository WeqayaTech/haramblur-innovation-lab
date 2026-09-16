#!/usr/bin/env python3
"""Anatomy of Gemini's 'unknown' answers in the Spotlight production run.

Streams verdicts_batch.jsonl once and tallies, per axis (gender / age), how the
unknown answers differ from the committed ones on every observable the run
recorded: person size, blur, SAM confidence, mask fragmentation, whether the
face is listed as visible, head covering, and the model's own confidence flag.
Read-only; single process; nothing here touches the training run.
"""
import json, sys, math
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/workspace/exp_unk/unk_stats.json"

SIZE_BANDS = [(0,32),(32,64),(64,96),(96,128),(128,192),(192,320),(320,10**9)]
def band(h):
    if h is None: return "na"
    for lo,hi in SIZE_BANDS:
        if lo <= h < hi: return f"{lo}-{hi if hi<10**9 else 'inf'}"
    return "na"

def blur_band(b):
    if b is None: return "na"
    for lo,hi in [(0,50),(50,150),(150,500),(500,2000),(2000,10**9)]:
        if lo <= b < hi: return f"{lo}-{hi if hi<10**9 else 'inf'}"
    return "na"

tot = 0
parse_bad = 0
verdict_c = Counter()
# groups: only real_person + depiction detections are meaningful for gender/age
g_c, a_c = Counter(), Counter()
joint = Counter()
# per-group observable tallies
stats = defaultdict(lambda: {"n":0, "h":[], "blur":[], "conf":[], "parts":Counter(),
                             "face":0, "back":0, "none_parts":0, "head":Counter(),
                             "lowconf":0, "samconf":[], "est_age":0})
def add(key, r, v):
    s = stats[key]
    s["n"] += 1
    h = r.get("person_px_height");  s["h"].append(h) if h is not None else None
    b = r.get("blurriness");        s["blur"].append(b) if b is not None else None
    sc = r.get("sam_conf");         s["samconf"].append(sc) if sc is not None else None
    s["parts"][min(int(r.get("n_parts") or 0), 5)] += 1
    ex = v.get("exposed_body_parts") or []
    if isinstance(ex, str): ex = [ex]
    ex = [str(e).lower() for e in ex]
    if "face" in ex: s["face"] += 1
    if "back" in ex: s["back"] += 1
    if ex == ["none"] or not ex: s["none_parts"] += 1
    s["head"][str(v.get("head_covering"))] += 1
    if str(v.get("confidence")) == "low": s["lowconf"] += 1
    if v.get("estimated_age") is not None: s["est_age"] += 1

# cross-tabs we want explicitly
xtab_size_g = defaultdict(Counter)   # size band -> gender
xtab_size_a = defaultdict(Counter)
xtab_face_g = defaultdict(Counter)   # face visible? -> gender
xtab_face_a = defaultdict(Counter)
xtab_head_g = defaultdict(Counter)
xtab_blur_g = defaultdict(Counter)
xtab_conf_g = defaultdict(Counter)   # model confidence -> gender
age_hist_unkg = Counter()            # estimated_age when gender unknown
xtab_agegrp_estage = defaultdict(Counter)

with open(PATH) as f:
    for line in f:
        tot += 1
        try: r = json.loads(line)
        except ValueError: parse_bad += 1; continue
        v = r.get("v") or {}
        if not r.get("parse_ok", True) or not v:
            parse_bad += 1; continue
        vd = str(v.get("verdict"))
        verdict_c[vd] += 1
        if vd not in ("real_person", "depiction"):
            continue
        g = str(v.get("gender")); a = str(v.get("age_group"))
        g_c[g] += 1; a_c[a] += 1; joint[(g,a)] += 1
        add(("gender", g), r, v)
        add(("age", a), r, v)
        h = r.get("person_px_height")
        ex = v.get("exposed_body_parts") or []
        if isinstance(ex, str): ex = [ex]
        ex = [str(e).lower() for e in ex]
        fk = "face_listed" if "face" in ex else ("back_listed" if "back" in ex else "no_face")
        xtab_size_g[band(h)][g] += 1
        xtab_size_a[band(h)][a] += 1
        xtab_face_g[fk][g] += 1
        xtab_face_a[fk][a] += 1
        xtab_head_g[str(v.get("head_covering"))][g] += 1
        xtab_blur_g[blur_band(r.get("blurriness"))][g] += 1
        xtab_conf_g[str(v.get("confidence"))][g] += 1
        ea = v.get("estimated_age")
        if g == "unknown":
            age_hist_unkg["none" if ea is None else ("0" if ea==0 else
                ("1-12" if ea<=12 else ("13-17" if ea<18 else ("18-29" if ea<30 else "30+"))))] += 1
        if a == "unknown":
            xtab_agegrp_estage["unknown_age"]["none" if ea is None else ("0" if ea==0 else
                ("1-12" if ea<=12 else ("13-17" if ea<18 else "18+")))] += 1

def q(xs):
    if not xs: return {}
    xs = sorted(xs); n = len(xs)
    pick = lambda p: xs[min(n-1, int(p*n))]
    return {"n":n, "p10":round(pick(.10),2), "p25":round(pick(.25),2), "median":round(pick(.50),2),
            "p75":round(pick(.75),2), "p90":round(pick(.90),2), "mean":round(sum(xs)/n,2)}

out = {"file": PATH, "total_lines": tot, "unparsed_or_missing": parse_bad,
       "verdicts": dict(verdict_c), "gender": dict(g_c), "age_group": dict(a_c),
       "joint_gender_age": {f"{k[0]}|{k[1]}": v for k,v in joint.items()},
       "groups": {}, "xtab": {
           "size_band_x_gender": {k: dict(v) for k,v in xtab_size_g.items()},
           "size_band_x_age":    {k: dict(v) for k,v in xtab_size_a.items()},
           "face_x_gender":      {k: dict(v) for k,v in xtab_face_g.items()},
           "face_x_age":         {k: dict(v) for k,v in xtab_face_a.items()},
           "head_covering_x_gender": {k: dict(v) for k,v in xtab_head_g.items()},
           "blur_band_x_gender": {k: dict(v) for k,v in xtab_blur_g.items()},
           "model_confidence_x_gender": {k: dict(v) for k,v in xtab_conf_g.items()},
           "est_age_when_gender_unknown": dict(age_hist_unkg),
           "est_age_when_age_unknown": {k: dict(v) for k,v in xtab_agegrp_estage.items()},
       }}
for (axis, val), s in stats.items():
    out["groups"][f"{axis}:{val}"] = {
        "n": s["n"],
        "person_px_height": q(s["h"]), "blurriness": q(s["blur"]), "sam_conf": q(s["samconf"]),
        "n_parts": dict(s["parts"]),
        "face_listed_pct": round(100*s["face"]/s["n"], 2) if s["n"] else None,
        "back_listed_pct": round(100*s["back"]/s["n"], 2) if s["n"] else None,
        "no_exposed_parts_pct": round(100*s["none_parts"]/s["n"], 2) if s["n"] else None,
        "model_low_confidence_pct": round(100*s["lowconf"]/s["n"], 2) if s["n"] else None,
        "has_estimated_age_pct": round(100*s["est_age"]/s["n"], 2) if s["n"] else None,
        "head_covering": dict(s["head"].most_common(10)),
    }
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT,"w"), indent=1)
print(json.dumps({k: out[k] for k in ("total_lines","unparsed_or_missing","verdicts","gender","age_group")}, indent=1))
print("wrote", OUT)
