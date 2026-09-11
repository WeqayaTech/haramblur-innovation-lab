#!/usr/bin/env python3
"""Where do INT8 and fp32 predictions differ? y26n_humanshaped_v2 @640, Spotlight-val, conf 0.45.
Reuses map_eval.py's GT/ignore loaders and IoU so the matching is the same as the mAP numbers."""
import json, base64, io, random, sys, html
from pathlib import Path
sys.path.insert(0, "/workspace/data_inspection_tools/vlm-cluster")
from map_eval import iou, load_gt, load_ignore, _ioa
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CONF, MATCH, PAIR = 0.45, 0.5, 0.7
MODEL = sys.argv[1] if len(sys.argv) > 1 else "y26n_humanshaped_v2"
RAW32 = Path(f"/workspace/quant_matrix_calib500/{MODEL}/fp32tflite_sz640_spotval/raw")
RAW8 = Path(f"/workspace/quant_matrix_calib500/{MODEL}/sz640_spotval/raw")
GTD = Path("/workspace/spotlight/run/oiv7_val/labels"); IGD = Path("/workspace/spotlight/run/oiv7_val/ignore_unk")
IMG = Path("/workspace/exp12/images_val4232")
OUT = Path(f"/workspace/exports/int8_vs_fp32_{MODEL}_640.html")
NAMES = {0: "Woman", 1: "Man", 2: "Child"}
random.seed(0)

def load(raw):
    out = {}
    for p in sorted(raw.glob("*.json")):
        d = json.loads(p.read_text()); w, h = d["width"], d["height"]
        dets = [(int(r["cls"]), float(r["conf"]), (r["box_xyxy"][0]/w, r["box_xyxy"][1]/h, r["box_xyxy"][2]/w, r["box_xyxy"][3]/h))
                for r in d["detections"] if not r.get("excluded") and float(r["conf"]) >= CONF]
        out[p.stem] = (w, h, dets)
    return out

A, B = load(RAW32), load(RAW8)
stems = sorted(set(A) & set(B))
gt, ign = load_gt(GTD, stems), load_ignore(IGD, stems)

def match(dets, gboxes, ignb):
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1])
    used = [False]*len(gboxes); dstat = [None]*len(dets); ghit = [None]*len(gboxes)
    for i in order:
        c, conf, b = dets[i]; best, bi = 0.0, -1
        for j, (gc, gb) in enumerate(gboxes):
            if used[j] or gc != c: continue
            v = iou(b, gb)
            if v > best: best, bi = v, j
        if best >= MATCH: used[bi] = True; dstat[i] = ("TP", bi, best); ghit[bi] = (i, best)
        elif any(_ioa(list(b), g) >= 0.5 for g in ignb): dstat[i] = ("IGN", None, 0.0)
        else:
            on, oj = max(((iou(b, g), j) for j, (gc, g) in enumerate(gboxes)), default=(0.0, -1))
            if on >= MATCH: dstat[i] = ("DUP" if gboxes[oj][0] == c else "DUPCLS", oj, on)   # extra box on a person already boxed
            else: dstat[i] = ("FP", None, 0.0)                                                  # on nothing
    return dstat, ghit

S = dict(n_img=len(stems), n_gt=0, both=0, a_only=0, b_only=0, neither=0, boxesA=0, boxesB=0, fpA=0, fpB=0,
         fpB_new=0, fpA_new=0, fp_shared=0, dupclsA=0, dupclsB=0, dupA=0, dupB=0, gt_dup_pairs=0, flips=0, diou=[], dconf=[], identical=0)
cands = dict(loose=[], missed=[], rescued=[], newfp=[], flip=[], sharedfp=[], same=[])
for s in stems:
    w, h, dA = A[s]; _, _, dB = B[s]
    gb = [(c, tuple(b)) for c in gt.get(s, {}) for b in gt[s][c]]
    ib = ign.get(s, [])
    stA, ghA = match(dA, gb, ib); stB, ghB = match(dB, gb, ib)
    S["n_gt"] += len(gb); S["boxesA"] += len(dA); S["boxesB"] += len(dB)
    fA = [i for i, st in enumerate(stA) if st[0] == "FP"]; fB = [i for i, st in enumerate(stB) if st[0] == "FP"]
    S["fpA"] += len(fA); S["fpB"] += len(fB)
    S["dupclsA"] += sum(st[0] == "DUPCLS" for st in stA); S["dupclsB"] += sum(st[0] == "DUPCLS" for st in stB)
    S["dupA"] += sum(st[0] == "DUP" for st in stA); S["dupB"] += sum(st[0] == "DUP" for st in stB)
    S["gt_dup_pairs"] += sum(1 for i in range(len(gb)) for j in range(i + 1, len(gb)) if iou(gb[i][1], gb[j][1]) >= MATCH)
    for i in fA:
        for k in fB:
            if iou(dA[i][2], dB[k][2]) >= MATCH:
                S["fp_shared"] += 1; cands["sharedfp"].append((max(dA[i][1], dB[k][1]), s, i)); break
    diffs = []
    for j, (gc, g) in enumerate(gb):
        ha, hb = ghA[j], ghB[j]
        if ha and hb:
            S["both"] += 1; d = hb[1] - ha[1]; S["diou"].append(d); S.setdefault("iouA", []).append(ha[1]); S.setdefault("iouB", []).append(hb[1])
            if d <= -0.15: diffs.append(("loose", -d, j))
        elif ha: S["a_only"] += 1; diffs.append(("missed", dA[ha[0]][1], j))
        elif hb: S["b_only"] += 1; diffs.append(("rescued", dB[hb[0]][1], j))
        else: S["neither"] += 1
    for i in fB:
        if all(iou(dB[i][2], dA[k][2]) < MATCH for k in range(len(dA))): S["fpB_new"] += 1; diffs.append(("newfp", dB[i][1], i))
    for i in fA:
        if all(iou(dA[i][2], dB[k][2]) < MATCH for k in range(len(dB))): S["fpA_new"] += 1
    for i in range(len(dA)):
        for k in range(len(dB)):
            if iou(dA[i][2], dB[k][2]) >= PAIR:
                S["dconf"].append(abs(dA[i][1] - dB[k][1]))
                if dA[i][0] != dB[k][0]:
                    S["flips"] += 1; diffs.append(("flip", max(dA[i][1], dB[k][1]), i))
                    key = f"{NAMES[dA[i][0]]} → {NAMES[dB[k][0]]}"
                    who = "fp32 matches GT" if stA[i][0] == "TP" else ("INT8 matches GT" if stB[k][0] == "TP" else "neither matches a labelled person")
                    S.setdefault("flipmat", {}); S["flipmat"][key] = S["flipmat"].get(key, 0) + 1
                    S.setdefault("flipwho", {}); S["flipwho"][who] = S["flipwho"].get(who, 0) + 1
    if not diffs:
        S["identical"] += 1; cands["same"].append((0, s, None))
    for kind, score, ref in diffs: cands[kind].append((score, s, ref))

def draw(s, note_ref=None):
    w, h, dA = A[s]; _, _, dB = B[s]
    im = Image.open(IMG / f"{s}.jpg").convert("RGB"); W, H = im.size
    # redact women: GT Woman boxes, any Woman prediction from either arm, and unknown-gender regions
    for b in list(gt.get(s, {}).get(0, [])) + [b for c, _, b in dA if c == 0] + [b for c, _, b in dB if c == 0] + list(ign.get(s, [])):
        x1, y1, x2, y2 = b[0]*W, b[1]*H, b[2]*W, b[3]*H
        pw, ph = (x2 - x1) * 0.08, (y2 - y1) * 0.08
        bx = (int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(W, x2 + pw)), int(min(H, y2 + ph)))
        if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2: continue
        rad = max(14, int(max(bx[2] - bx[0], bx[3] - bx[1]) / 10))
        im.paste(im.crop(bx).filter(ImageFilter.GaussianBlur(rad)), bx)
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, W // 60))
    except Exception: font = ImageFont.load_default()
    def box(b, col, wd, txt=None):
        x1, y1, x2, y2 = b[0]*W, b[1]*H, b[2]*W, b[3]*H
        dr.rectangle([x1, y1, x2, y2], outline=col, width=wd)
        if txt:
            tw = dr.textlength(txt, font=font) if hasattr(dr, "textlength") else 60
            dr.rectangle([x1, max(0, y1-font.size-4), x1+tw+6, max(0, y1-font.size-4)+font.size+4], fill=col)
            dr.text((x1+3, max(0, y1-font.size-2)), txt, fill="white", font=font)
    for j, (c, g) in enumerate([(c, g) for c in gt.get(s, {}) for g in gt[s][c]]):
        box(g, (240, 240, 240), 2, f"#{j+1} GT {NAMES[c]}")
    for c, conf, b in dA: box(b, (42, 120, 214), 3, f"fp32 {NAMES[c]} {conf:.2f}")
    for c, conf, b in dB: box(b, (235, 104, 52), 3, f"int8 {NAMES[c]} {conf:.2f}")
    scale = 720 / max(W, H)
    if scale < 1: im = im.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()

def readout(s):
    """Per-image table: each GT person → what fp32 and INT8 said; then extra boxes."""
    w, h, dA = A[s]; _, _, dB = B[s]
    gb = [(c, tuple(b)) for c in gt.get(s, {}) for b in gt[s][c]]; ib = ign.get(s, [])
    stA, ghA = match(dA, gb, ib); stB, ghB = match(dB, gb, ib)
    def cell(dets, hit):
        if not hit: return '<td class="miss">missed</td>'
        i, v = hit; c, conf, _ = dets[i]
        return f'<td>{NAMES[c]} <b>{conf:.2f}</b> <span class="iou">IoU {v:.2f}</span></td>'
    dupof = {}
    for i in range(len(gb)):
        for j in range(i + 1, len(gb)):
            if iou(gb[i][1], gb[j][1]) >= MATCH: dupof.setdefault(j, i)
    rows = [f'<tr><td>#{j+1}</td><td>{NAMES[gc]}{f" <span class=iou>(label overlaps #{dupof[j]+1})</span>" if j in dupof else ""}</td>{cell(dA, ghA[j])}{cell(dB, ghB[j])}</tr>' for j, (gc, _) in enumerate(gb)]
    def tag(st):
        k, j, _ = st
        if k == "DUPCLS": return f"2nd box on #{j+1} (GT {NAMES[gb[j][0]]})"
        if k == "DUP": return f"duplicate of #{j+1}"
        return "unk-gender region" if k == "IGN" else "on nothing"
    def extra(dets, st):
        return [(i, dets[i][0], dets[i][1], tag(st[i]), st[i]) for i in range(len(dets)) if st[i][0] != "TP"]
    exA, exB = extra(dA, stA), extra(dB, stB); usedB = set()
    def ex(e): return f'<td>{NAMES[e[1]]} <b>{e[2]:.2f}</b> <span class="iou">{e[3]}</span></td>'
    def gtcell(e): return f'<td class="miss">→ #{e[4][1]+1}</td>' if e[4][0] in ("DUP", "DUPCLS") else '<td class="miss">no GT</td>'
    for e in exA:
        mate = next((f for f in exB if f[0] not in usedB and iou(dA[e[0]][2], dB[f[0]][2]) >= MATCH), None)
        if mate: usedB.add(mate[0])
        rows.append(f'<tr><td>–</td>{gtcell(e)}{ex(e)}{ex(mate) if mate else "<td class=miss>–</td>"}</tr>')
    for f in exB:
        if f[0] not in usedB: rows.append(f'<tr><td>–</td>{gtcell(f)}<td class="miss">–</td>{ex(f)}</tr>')
    return '<table class="ro"><tr><th>#</th><th>GT</th><th>fp32</th><th>INT8</th></tr>' + "".join(rows) + "</table>"

def pick(kind, n):
    c = cands[kind]
    if kind == "same": random.shuffle(c); return [x[1] for x in c[:n]]
    seen, out = set(), []
    for score, s, ref in sorted(c, key=lambda x: -x[0]):
        if s not in seen: seen.add(s); out.append(s)
        if len(out) >= n: break
    return out

import statistics as st
diou = S["diou"]; dconf = S["dconf"]
def pct(k, n): return f"{100*k/n:.1f}%" if n else "–"
hist = {}
for d in diou:
    k = ("≤ −0.20" if d <= -0.2 else "−0.20…−0.10" if d <= -0.1 else "−0.10…−0.05" if d <= -0.05 else "−0.05…+0.05" if d < 0.05 else "≥ +0.05")
    hist[k] = hist.get(k, 0) + 1
order = ["≤ −0.20", "−0.20…−0.10", "−0.10…−0.05", "−0.05…+0.05", "≥ +0.05"]

sections = [
  ("loose", "INT8 box noticeably looser on the same person", "Both arms found the person; INT8's box overlaps the ground truth by ≥0.15 IoU less than fp32's. This is the mAP50-95 story.", 8),
  ("missed", "fp32 found a person INT8 missed", "Ground-truth person matched by fp32 at conf ≥ 0.45 but not by INT8 — the escape direction.", 6),
  ("rescued", "INT8 found a person fp32 missed", "The reverse: INT8 matched a ground-truth person fp32 did not.", 6),
  ("newfp", "INT8 box on nothing, with no fp32 counterpart", "An INT8 box that is on no labelled person, no unknown-gender region, and is not a second-class box on a person already found — and that fp32 did not produce.", 6),
  ("flip", "Class disagreement on the same box", "fp32 and INT8 boxes overlap (IoU ≥ 0.7) but disagree on Woman / Man / Child.", 6),
  ("sharedfp", "Both models box something with no labelled person", "A fp32 box and an INT8 box on the same spot (IoU ≥ 0.5), on neither a labelled nor an unknown-gender person — a shared false positive, so not a quantization effect. Labels are Spotlight machine labels; a few of these may be real people the labeler missed.", 6),
  ("same", "Control: images with no difference at all", "Random images where fp32 and INT8 agree on every person and produce no extra boxes.", 4),
]
cards = []
for kind, title, desc, n in sections:
    items = pick(kind, n)
    cards.append(f'<section><h2>{html.escape(title)} <span class="cnt">{len(cands[kind])} images</span></h2><p class="desc">{html.escape(desc)}</p><div class="grid">'
                 + "".join(f'<figure><img src="data:image/jpeg;base64,{draw(s)}" alt=""><figcaption>{html.escape(s)}</figcaption>{readout(s)}</figure>' for s in items) + "</div></section>")

page = f"""<title>INT8 vs fp32 · {MODEL} @640</title>
<style>
:root{{--bg:#F6F7F9;--surface:#fff;--ink:#1B1F26;--ink2:#4B5563;--muted:#6B7280;--rule:#D9DEE5;--fp32:#2a78d6;--int8:#eb6834;--gt:#9aa0a6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#14171C;--surface:#1B1F26;--ink:#E8EAEE;--ink2:#B4BAC4;--muted:#8B93A1;--rule:#2E343D}}}}
:root[data-theme="dark"]{{--bg:#14171C;--surface:#1B1F26;--ink:#E8EAEE;--ink2:#B4BAC4;--muted:#8B93A1;--rule:#2E343D}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.5}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}} .sub{{color:var(--ink2);margin:0 0 18px;max-width:80ch}}
h2{{font-size:19px;margin:36px 0 4px}} .cnt{{font-size:13px;color:var(--muted);font-weight:500;margin-left:8px}} .desc{{color:var(--ink2);margin:0 0 12px;max-width:80ch}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 18px;font-size:13px}} .sw{{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:6px}}
table{{border-collapse:collapse;font-size:13.5px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;overflow:hidden}} th,td{{padding:6px 12px;text-align:right;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}} th{{color:var(--muted);font-weight:600;font-size:12px}} td:first-child,th:first-child{{text-align:left}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;align-items:start}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}} figure{{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:8px;overflow:hidden}} img{{display:block;width:100%;height:auto}} figcaption{{font:11.5px ui-monospace,monospace;color:var(--muted);padding:6px 10px}}
.note{{font-size:13px;color:var(--ink2);max-width:80ch}}
table.ro{{width:100%;border:0;border-radius:0;border-top:1px solid var(--rule);font:12px ui-monospace,SFMono-Regular,Menlo,monospace}} table.ro th,table.ro td{{padding:4px 8px;text-align:left;white-space:nowrap}} table.ro .miss{{color:var(--muted)}} table.ro .iou{{color:var(--muted);font-size:11px}}
</style>
<div class="wrap">
<h1>Where INT8 and fp32 disagree — {MODEL} @640</h1>
<p class="sub">Same {S['n_img']:,} Spotlight-val images, both models' boxes at the production threshold conf ≥ {CONF}. fp32 = fp32 TFLite export; INT8 = train-calibrated INT8 TFLite. Matching is class-aware greedy IoU ≥ {MATCH} against ground truth, identical to the mAP scorer. <b>Women are redacted</b> in every image: ground-truth Woman boxes, any Woman prediction from either model, and unknown-gender regions are blurred before the boxes are drawn. Note: a model can emit two boxes of different classes on one person (NMS runs per class); those are counted as "2nd box on #n", not as false positives, and the readout under each image points every extra box at the person it sits on.</p>
<div class="legend"><span><span class="sw" style="background:var(--gt)"></span>ground truth</span><span><span class="sw" style="background:var(--fp32)"></span>fp32</span><span><span class="sw" style="background:var(--int8)"></span>INT8</span></div>
<div class="stats">
<table><tr><th>ground-truth people ({S['n_gt']:,})</th><th>count</th><th>share</th></tr>
<tr><td>found by both</td><td>{S['both']:,}</td><td>{pct(S['both'],S['n_gt'])}</td></tr>
<tr><td>found by fp32 only (INT8 missed)</td><td>{S['a_only']:,}</td><td>{pct(S['a_only'],S['n_gt'])}</td></tr>
<tr><td>found by INT8 only (INT8 rescued)</td><td>{S['b_only']:,}</td><td>{pct(S['b_only'],S['n_gt'])}</td></tr>
<tr><td>found by neither</td><td>{S['neither']:,}</td><td>{pct(S['neither'],S['n_gt'])}</td></tr></table>
<table><tr><th>boxes at conf ≥ {CONF}</th><th>fp32</th><th>INT8</th></tr>
<tr><td>total boxes</td><td>{S['boxesA']:,}</td><td>{S['boxesB']:,}</td></tr>
<tr><td>2nd box of another class on a labelled person</td><td>{S['dupclsA']:,}</td><td>{S['dupclsB']:,}</td></tr>
<tr><td>same-class duplicate box on a labelled person</td><td>{S['dupA']:,}</td><td>{S['dupB']:,}</td></tr>
<tr><td>boxes on nothing (true false positives)</td><td>{S['fpA']:,}</td><td>{S['fpB']:,}</td></tr>
<tr><td>… with no counterpart in the other arm</td><td>{S['fpA_new']:,}</td><td>{S['fpB_new']:,}</td></tr>
<tr><td>… shared by both arms (same phantom, IoU ≥ {MATCH})</td><td colspan="2">{S['fp_shared']:,}</td></tr>
<tr><td>ground-truth boxes overlapping another GT box (label duplicates)</td><td colspan="2">{S['gt_dup_pairs']:,} pairs</td></tr>
<tr><td>class flips on overlapping boxes (IoU ≥ {PAIR})</td><td colspan="2">{S['flips']:,}</td></tr>
<tr><td>images with no difference at all</td><td colspan="2">{S['identical']:,} / {S['n_img']:,} ({pct(S['identical'],S['n_img'])})</td></tr></table>
<table><tr><th>people found by both: IoU(INT8) − IoU(fp32)</th><th>count</th><th>share</th></tr>
{"".join(f"<tr><td>{k}</td><td>{hist.get(k,0):,}</td><td>{pct(hist.get(k,0),len(diou))}</td></tr>" for k in order)}
<tr><td>mean IoU to GT: fp32 / INT8</td><td colspan="2">{st.mean(S.get('iouA',[0])):.3f} / {st.mean(S.get('iouB',[0])):.3f}</td></tr></table>
</div>
<h2>Class flips <span class="cnt">{S['flips']:,} overlapping box pairs</span></h2>
<p class="desc">fp32 and INT8 boxes overlap (IoU ≥ {PAIR}) but carry different classes. Direction and who agrees with the human-verified label:</p>
<div class="stats"><table><tr><th>fp32 → INT8</th><th>count</th></tr>{"".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>" for k, v in sorted(S.get('flipmat', {}).items(), key=lambda kv: -kv[1]))}</table>
<table><tr><th>who is right</th><th>count</th></tr>{"".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>" for k, v in sorted(S.get('flipwho', {}).items(), key=lambda kv: -kv[1]))}</table></div>
<p class="note">Mean |Δconf| on overlapping boxes: {st.mean(dconf) if dconf else 0:.3f} (median {st.median(dconf) if dconf else 0:.3f}, n={len(dconf):,}). Mean IoU-to-GT delta on people found by both: {st.mean(diou) if diou else 0:+.4f} (median {st.median(diou) if diou else 0:+.4f}).</p>
{"".join(cards)}
</div>"""
OUT.write_text(page)
print("WROTE", OUT, OUT.stat().st_size, "bytes")
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in S.items()}))
print("HIST", hist)
