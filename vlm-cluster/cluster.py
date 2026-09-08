#!/usr/bin/env python3
"""
HARAMBLUR VLM-cluster — Stage 2: embed, cluster, report.

Reads descriptions.jsonl from stage 1, renders each object's structured
description to a canonical text, embeds it, finds NATURAL clusters (HDBSCAN),
and reports cluster sizes + a 2D map. Small clusters and the noise/outlier
region are the under-represented slices ("babies with chubby faces",
"man in a crowd") — the data you may need more of.

Cheap and re-runnable; tune freely without touching the VLM stage.

    python cluster.py --in ./run1 --out ./run1/clusters

Fallbacks (so it runs anywhere): embeddings sentence-transformers -> mock;
reduction UMAP -> PCA; clustering HDBSCAN -> KMeans.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

# fields that define a "slice" — used for cluster labels and canonical text
KEY_FIELDS = ["class", "apparent_age_band", "orientation", "visible_part",
              "clothing_coverage", "setting", "scene_type", "crowd"]

# default fields for FEATURE-VALUE clustering (one-hot, not text embeddings)
FEATURE_FIELDS = [
    "apparent_age_band", "apparent_attire_region", "facial_hair", "hair_length",
    "head_covering", "garment_type", "clothing_coverage", "build",
    "orientation", "visible_part", "skin_tone_mst", "crowd", "setting",
]


def onehot(recs, fields):
    """Build a one-hot feature matrix from categorical field values.
    Clustering on this groups records by shared feature values (apparent age,
    attire, facial hair, ...) instead of by blurry text-embedding similarity."""
    values = {f: sorted({flat(r).get(f, "unknown") for r in recs}) for f in fields}
    cols = [(f, v) for f in fields for v in values[f]]
    idx = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(recs), len(cols)), dtype=np.float32)
    for i, r in enumerate(recs):
        fr = flat(r)
        for f in fields:
            X[i, idx[(f, fr.get(f, "unknown"))]] = 1.0
    return X, cols


def load_records(in_dirs, pattern="descriptions*.jsonl"):
    # picks up single-run (descriptions.jsonl) and sharded
    # (descriptions.shard0.jsonl, ...) outputs across one or more run dirs.
    # pattern can be overridden (e.g. "compare*.jsonl" for compare_child.py output)
    files = []
    for d in in_dirs:
        files += sorted(Path(d).glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern!r} in {in_dirs}")
    seen, recs = set(), []
    for jf in files:
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:                      # tolerate a half-written last line
                r = json.loads(line)   # (describe.py may be appending live)
            except json.JSONDecodeError:
                continue
            if r["id"] in seen:        # dedupe across runs / shards / reruns
                continue
            seen.add(r["id"])
            recs.append(r)
    print(f"[cluster] loaded {len(recs)} unique objects from "
          f"{len(files)} file(s) across {len(in_dirs)} dir(s)")
    return recs


def flat(rec: dict) -> dict:
    """Merge class + object + scene into one flat dict of strings."""
    d = {"class": rec.get("class", "unknown")}
    d.update(rec.get("object", {}))
    d.update(rec.get("scene", {}))
    return d


def canonical_text(rec: dict) -> str:
    """Stable text rendering for embedding — meaning over phrasing."""
    f = flat(rec)
    parts = [f"a {f.get('class','person')}"]
    for k in ["apparent_age_band", "build", "facial_hair", "hair_length",
              "head_covering", "garment_type", "apparent_attire_region",
              "clothing", "clothing_coverage", "pose", "orientation",
              "occlusion", "visible_part", "activity", "skin_tone_mst"]:
        v = f.get(k)
        if v and v != "unknown" and v != "none":
            parts.append(f"{k.replace('_',' ')}: {v}")
    for k in ["setting", "scene_type", "crowd", "short_caption"]:
        v = f.get(k)
        if v and v != "unknown":
            parts.append(f"{k.replace('_',' ')}: {v}")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def embed(texts, mode, model_name):
    if mode == "mock":
        # deterministic hash-based vectors — for local plumbing tests only
        rng = np.random.default_rng(0)
        basis = {}
        vecs = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for tok in t.split():
                if tok not in basis:
                    basis[tok] = rng.standard_normal(64).astype(np.float32)
                v += basis[tok]
            n = np.linalg.norm(v) or 1.0
            vecs.append(v / n)
        return np.vstack(vecs)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(texts, show_progress_bar=True,
                                   normalize_embeddings=True), dtype=np.float32)


def reduce(X, n_components, min_dist=0.0, metric="cosine"):
    """UMAP reduction. Clustering happens in LOW-dim space (not raw 384-D),
    otherwise HDBSCAN's density estimate collapses and labels everything noise."""
    try:
        import umap
        nn = min(15, max(2, len(X) - 1))
        return umap.UMAP(n_components=n_components, n_neighbors=nn,
                         min_dist=min_dist, metric=metric,
                         random_state=42).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        print(f"[cluster] UMAP unavailable -> PCA({n_components})")
        return PCA(n_components=n_components, random_state=42).fit_transform(X)


def cluster(Xr, min_cluster_size, min_samples, k_fallback):
    try:
        import hdbscan
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,           # lower -> less aggressive noise
            metric="euclidean").fit_predict(Xr)
        return labels, "hdbscan"
    except Exception:
        from sklearn.cluster import KMeans
        k = min(k_fallback, max(2, len(Xr) // max(1, min_cluster_size)))
        print(f"[cluster] HDBSCAN unavailable -> KMeans(k={k})")
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xr), "kmeans"


# --------------------------------------------------------------------------
# Labeling & reporting
# --------------------------------------------------------------------------

def cluster_label(recs_in_cluster):
    """Human-readable label from the modal values of key fields."""
    parts = []
    for field in ["apparent_age_band", "class", "facial_hair", "garment_type",
                  "orientation", "visible_part", "clothing_coverage",
                  "scene_type", "crowd"]:
        vals = Counter()
        for r in recs_in_cluster:
            v = flat(r).get(field, "unknown")
            if v not in ("unknown", "none", ""):
                vals[v] += 1
        if vals:
            top, cnt = vals.most_common(1)[0]
            if cnt >= 0.4 * len(recs_in_cluster):  # only if reasonably dominant
                parts.append(top.replace("_", " "))
    return ", ".join(dict.fromkeys(parts)) or "mixed"


def top_fields(recs, field, n=3):
    c = Counter(flat(r).get(field, "unknown") for r in recs)
    total = sum(c.values()) or 1
    return ", ".join(f"{v} {100*k/total:.0f}%" for v, k in c.most_common(n))


def _wrap(text, max_w, font, scale, thick):
    import cv2
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if cv2.getTextSize(trial, font, scale, thick)[0][0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def montage(ids, crops_dirs, thumb=150, cols=6, title=None):
    import cv2
    imgs = []
    for oid in ids:
        p = next((cd / f"{oid}.jpg" for cd in crops_dirs
                  if (cd / f"{oid}.jpg").exists()), None)
        if p is not None:
            im = cv2.imread(str(p))
            if im is not None:
                s = thumb / max(im.shape[:2])
                im = cv2.resize(im, (max(1, int(im.shape[1]*s)),
                                     max(1, int(im.shape[0]*s))))
                canvas = np.full((thumb, thumb, 3), 30, np.uint8)
                y0 = (thumb - im.shape[0]) // 2
                x0 = (thumb - im.shape[1]) // 2
                canvas[y0:y0+im.shape[0], x0:x0+im.shape[1]] = im
                imgs.append(canvas)
    if not imgs:
        return None
    cols = min(cols, len(imgs))
    rows = math.ceil(len(imgs) / cols)
    grid_w = cols * thumb

    # optional title banner across the top
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    lines = _wrap(title, grid_w - 16, font, scale, thick) if title else []
    banner = (10 + 22 * len(lines) + 6) if lines else 0

    sheet = np.full((rows * thumb + banner, grid_w, 3), 30, np.uint8)
    y = 24
    for ln in lines:
        cv2.putText(sheet, ln, (8, y), font, scale, (235, 235, 235), thick,
                    cv2.LINE_AA)
        y += 22
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[banner + r*thumb:banner + (r+1)*thumb, c*thumb:(c+1)*thumb] = im
    return sheet


def scatter(emb2d, labels, path, names=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(11, 8))
    uniq = sorted(set(labels.tolist()))
    for lab in uniq:
        m = labels == lab
        if lab == -1:
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, c="#cccccc", alpha=0.4)
        else:
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=10, alpha=0.6)
    # name each cluster at its centroid (median is robust to outliers)
    if names:
        for lab in uniq:
            if lab == -1:
                continue
            m = labels == lab
            cx, cy = float(np.median(emb2d[m, 0])), float(np.median(emb2d[m, 1]))
            ax.text(cx, cy, f"c{lab}: {names.get(lab, '')}", fontsize=8,
                    ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888888", alpha=0.85))
    ax.set_title("Clusters (each labeled by its top features)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="VLM-cluster stage 2: cluster")
    ap.add_argument("--in", dest="indir", required=True, nargs="+",
                    help="one or more stage-1 out dirs (e.g. ./run1 ./run_full)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["features", "text"], default="features",
                    help="features=one-hot of field values (recommended); "
                         "text=embed the description sentence")
    ap.add_argument("--features", nargs="+", default=FEATURE_FIELDS,
                    help="fields to cluster on in feature mode")
    ap.add_argument("--k", type=int, default=12,
                    help="number of KMeans clusters in feature mode (0=use HDBSCAN)")
    ap.add_argument("--embed", choices=["sbert", "mock"], default="sbert")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--min-samples", type=int, default=5,
                    help="HDBSCAN min_samples; lower = less noise, looser clusters")
    ap.add_argument("--reduce-dims", type=int, default=5,
                    help="UMAP dims for clustering (NOT raw 384-D)")
    ap.add_argument("--k-fallback", type=int, default=12)
    ap.add_argument("--montage-samples", type=int, default=12)
    ap.add_argument("--small-frac", type=float, default=0.02,
                    help="clusters below this fraction are flagged as gaps")
    ap.add_argument("--pattern", default="descriptions*.jsonl",
                    help="glob for input files (e.g. 'compare*.jsonl' for "
                         "compare_child.py output)")
    ap.add_argument("--where", nargs="+", default=None,
                    help="filter records before clustering: field=value pairs "
                         "(ANDed), matched against top-level or object fields, "
                         "e.g. --where category=model_error")
    ap.add_argument("--crops-dir", nargs="+", default=None,
                    help="override crop directories (default: <indir>/crops "
                         "for each --in; use this when jsonl and crops live in "
                         "different dirs, e.g. compare_child.py output)")
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.indir]
    if args.out:
        out = Path(args.out)
    else:
        out = (in_dirs[0] / "clusters" if len(in_dirs) == 1
               else Path("./clusters_all"))
    out.mkdir(parents=True, exist_ok=True)
    crops_dirs = ([Path(c) for c in args.crops_dir] if args.crops_dir
                  else [d / "crops" for d in in_dirs])

    recs = load_records(in_dirs, pattern=args.pattern)

    if args.where:
        conds = [w.split("=", 1) for w in args.where]
        before = len(recs)
        recs = [r for r in recs if all(
            str(r.get(field, r.get("object", {}).get(field))) == val
            for field, val in conds)]
        print(f"[cluster] --where {args.where}: {len(recs)} of {before} records kept")
        if not recs:
            raise SystemExit("no records matched --where filter")

    texts = [canonical_text(r) for r in recs]  # used for objects.csv (+ text mode)

    if args.mode == "features":
        X, cols = onehot(recs, args.features)
        print(f"[cluster] feature mode: {X.shape[1]} one-hot columns "
              f"from {len(args.features)} fields")
        if args.k > 0:
            from sklearn.cluster import KMeans
            k = min(args.k, max(2, len(recs)))
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            algo = f"kmeans(k={k}) on features"
        else:  # auto: HDBSCAN on reduced one-hot
            Xr = reduce(X, args.reduce_dims, min_dist=0.0)
            labels, algo = cluster(Xr, args.min_cluster_size, args.min_samples,
                                   args.k_fallback)
        emb2d = reduce(X, 2, min_dist=0.1)
    else:
        X = embed(texts, args.embed, args.embed_model)
        Xr = reduce(X, args.reduce_dims, min_dist=0.0)
        labels, algo = cluster(Xr, args.min_cluster_size, args.min_samples,
                               args.k_fallback)
        emb2d = Xr[:, :2] if args.reduce_dims == 2 else reduce(X, 2, min_dist=0.1)

    labels = np.asarray(labels)
    n_noise = int(np.sum(labels == -1))
    print(f"[cluster] {algo}: {len(set(labels.tolist()) - {-1})} clusters "
          f"({n_noise} noise, {100*n_noise/len(labels):.0f}%)")

    # group records by cluster
    by_cluster = {}
    for r, lab, txt in zip(recs, labels, texts):
        r["_cluster"] = int(lab)
        by_cluster.setdefault(int(lab), []).append(r)

    total = len(recs)
    rows = []
    for lab, members in by_cluster.items():
        rows.append({
            "cluster": lab,
            "label": "noise / outliers" if lab == -1 else cluster_label(members),
            "size": len(members),
            "pct": 100 * len(members) / total,
            "age": top_fields(members, "apparent_age_band"),
            "scene": top_fields(members, "scene_type"),
            "coverage": top_fields(members, "clothing_coverage"),
            "members": members,
        })
    rows.sort(key=lambda r: r["size"])  # smallest first = the gaps

    # outputs --------------------------------------------------------------
    # short name per cluster (top 2 features) for the 2D map
    short = {r["cluster"]: ", ".join(r["label"].split(", ")[:2]) for r in rows}
    scatter(emb2d, np.asarray(labels), out / "cluster_map.png", names=short)

    mont_dir = out / "montages"
    mont_dir.mkdir(exist_ok=True)
    try:
        import cv2
        for r in rows:
            ids = [m["id"] for m in r["members"][:args.montage_samples]]
            title = f"c{r['cluster']}: {r['label']}  (n={r['size']}, {r['pct']:.1f}%)"
            sheet = montage(ids, crops_dirs, title=title)
            if sheet is not None:
                cv2.imwrite(str(mont_dir / f"cluster_{r['cluster']}.jpg"), sheet)
    except ImportError:
        pass

    # csv (per object)
    import csv
    with (out / "objects.csv").open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["id", "cluster", "class", "canonical_text"])
        for r, txt in zip(recs, texts):
            w.writerow([r["id"], r["_cluster"],
                       r.get("class", r.get("label", "unknown")), txt])

    # markdown summary
    lines = ["# HARAMBLUR — VLM description clusters\n",
             f"- Objects: **{total}**  |  Clusters: **{len(by_cluster)-(1 if -1 in by_cluster else 0)}**"
             f"  |  Algo: **{algo}**",
             f"- Map: ![map](cluster_map.png)\n",
             "## Clusters (smallest first = candidate gaps)\n",
             "| # | label | size | % | age | scene | coverage |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " ⚠️" if (r["cluster"] != -1 and r["pct"] < args.small_frac*100) else ""
        lines.append(f"| {r['cluster']} | {r['label']}{flag} | {r['size']} | "
                     f"{r['pct']:.1f}% | {r['age']} | {r['scene']} | {r['coverage']} |")
    lines.append("\n## Under-represented slices (get more data here)\n")
    for r in rows:
        if r["cluster"] == -1:
            continue
        if r["pct"] < args.small_frac * 100:
            lines.append(f"- **{r['label']}** — {r['size']} objs ({r['pct']:.1f}%) "
                         f"· see montages/cluster_{r['cluster']}.jpg")
    lines.append("\n## Cluster montages\n")
    for r in rows:
        lines.append(f"### c{r['cluster']} — {r['label']} ({r['size']}, {r['pct']:.1f}%)")
        lines.append(f"![c{r['cluster']}](montages/cluster_{r['cluster']}.jpg)\n")
    (out / "clusters.md").write_text("\n".join(lines), encoding="utf-8")

    json.dump(
        {"total": total, "algo": algo,
         "clusters": [{k: r[k] for k in
                       ["cluster", "label", "size", "pct", "age", "scene", "coverage"]}
                      for r in rows]},
        (out / "clusters.json").open("w"), indent=2)

    print(f"[cluster] report -> {out/'clusters.md'}")


if __name__ == "__main__":
    main()
