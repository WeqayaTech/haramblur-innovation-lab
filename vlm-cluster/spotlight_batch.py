#!/usr/bin/env python3
"""
SPOTLIGHT — Batch API path (half the price of live calls, async).

Live calls bill immediately, so the runner can stop mid-stream at a budget.
Batch is submit-then-collect, so the budget is enforced at SUBMIT time: we
select exactly as many pending detections as the budget buys, send them, and
never submit more until you ask.

  submit   pick pending detections up to --max-spend, build the spotlight
           crops, write a batch request file, upload it, start the job, and
           record which detections are in flight (so a later submit or a live
           run never redoes them)
  poll     job status
  collect  download finished results, append them to verdicts.jsonl in the
           SAME schema the live runner writes -> emit / verify_labels /
           trace_report all work unchanged
  status   what is done, in flight, and still pending

    python3 spotlight_batch.py submit --images <imgs> --raw-labels <raw> \\
        --out <run dir> --max-spend 2
    python3 spotlight_batch.py poll    --out <run dir>
    python3 spotlight_batch.py collect --out <run dir>
    python3 spotlight_batch.py status  --out <run dir>

    python3 spotlight_batch.py --selftest        # no network, no data

Bookkeeping lives in <out>/batch/:
    pending_<jobid>.json   the (image_stem, det_index) list submitted
    requests_<jobid>.jsonl the exact payload sent
    jobs.json              every job: id, state, counts, cost estimate
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

from PIL import Image

from spotlight_run import (IMG_EXTS, PROMPT, PROMPT_VERSION, build_crop,
                           load_raw, parse_verdict, blurriness)

# Batch API list price is 50% of standard (verified 2026-07-27).
BATCH_RATE_IN = 0.15      # $/1M input tokens
BATCH_RATE_OUT = 1.25     # $/1M output tokens
# Measured on the e1 prompt (EXP-2026-10 + production smoke): used only to
# decide how many detections a budget buys. Actual cost is recomputed from
# the usage metadata that comes back with the results.
EST_IN_TOK = 1160
EST_OUT_TOK = 100
# Each request embeds a base64 JPEG crop (~40 KB), so the request FILE size is
# the binding constraint, not the request count. 20k requests ~= 800 MB.
MAX_REQUESTS_PER_JOB = 20_000


def est_cost_per_det():
    return (EST_IN_TOK * BATCH_RATE_IN + EST_OUT_TOK * BATCH_RATE_OUT) / 1e6


class SubmitLock:
    """Cross-machine mutex on the shared volume. Two coordinators running
    submit at the same moment would both see the same pending set and buy the
    same detections twice; this serialises them. O_EXCL create is atomic on
    the network volume."""

    def __init__(self, out: Path, stale_minutes=30, name=""):
        # One lock per shard: sharded coordinators work disjoint image sets,
        # so they cannot buy the same detection and must NOT serialise.
        self.path = _bdir(out) / (f"submit{name}.lock")
        self.stale = stale_minutes * 60

    def __enter__(self):
        import os
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale:
                print(f"[lock] removing stale lock ({age/60:.0f} min old) "
                      f"from {self.path.read_text().strip()}")
                self.path.unlink(missing_ok=True)
            else:
                raise SystemExit(
                    f"[lock] another submit is running ({self.path.read_text().strip()}, "
                    f"{age/60:.1f} min ago). Wait, or delete {self.path} if that "
                    f"process is dead.")
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} pid={os.getpid()} "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _bdir(out: Path):
    d = out / "batch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jobs(out: Path):
    f = _bdir(out) / "jobs.json"
    return json.loads(f.read_text()) if f.exists() else []


def _save_jobs(out: Path, jobs):
    (_bdir(out) / "jobs.json").write_text(json.dumps(jobs, indent=2))


def done_keys(out: Path):
    """Detections that already have a verdict (live or collected batch)."""
    keys = set()
    for f in sorted(out.glob("verdicts*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["image_stem"], r["det_index"]))
    return keys


def inflight_keys(out: Path):
    """Detections submitted in a job that has not been collected yet — the
    guard that stops the same image being paid for twice.

    Scans pending_*.json files DIRECTLY rather than iterating jobs.json, so a
    job entry lost to a concurrent write still blocks resubmission. Fails
    safe: an unknown pending file means 'assume in flight', never 're-pay'."""
    collected = {j["short_id"] for j in _jobs(out) if j.get("collected")}
    keys = set()
    for pf in _bdir(out).glob("pending_*.json"):
        if pf.stem.replace("pending_", "") in collected:
            continue
        try:
            keys |= {tuple(k) for k in json.loads(pf.read_text())}
        except (ValueError, OSError):
            continue
    return keys


def pending(images_dir: Path, raw_dir: Path, out: Path, limit=None,
            shards=1, shard_index=0):
    """Detections with a sidecar, no verdict, and not already in flight.

    Building one request means opening the image, cropping, outlining the mask
    and base64-encoding it -- ~35 ms per detection over the network volume, so
    a single coordinator tops out near 10k detections per 7 minutes. `shards`
    splits the IMAGE list by stable hash so several machines can build crops
    concurrently on disjoint sets (same scheme as autolabel_sam_raw.py). With
    disjoint sets no two coordinators can select the same detection, which is
    why the submit lock is taken per shard rather than globally.
    """
    skip = done_keys(out) | inflight_keys(out)
    todo = []
    for p in sorted(q for q in images_dir.rglob("*") if q.suffix.lower() in IMG_EXTS):
        if shards > 1 and (int(hashlib.md5(str(p).encode()).hexdigest(), 16)
                           % shards != shard_index):
            continue
        rec = load_raw(raw_dir, p.stem)
        if rec is None:
            continue
        for i in range(len(rec.get("detections", []))):
            if (p.stem, i) not in skip:
                todo.append((p, p.stem, i))
                if limit and len(todo) >= limit:
                    return todo
    return todo


def _crop_b64(img: Image.Image, det):
    parts = [[(x, y) for x, y in part] for part in det.get("parts", [])]
    crop, _ = build_crop(img, det["box"], parts)
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), crop


def submit(images_dir: Path, raw_dir: Path, out: Path, max_spend, model,
           max_tokens, dry_run, max_requests=MAX_REQUESTS_PER_JOB,
           shards=1, shard_index=0):
    """One job per call. Size is bounded by BOTH the budget and a hard request
    cap, because each request carries a base64 crop (~40 KB) and batch input
    files have a size limit — 1.34M requests would be ~54 GB in one file."""
    per_det = est_cost_per_det()
    n_budget = int(max_spend / per_det) if max_spend else None
    limit = min(x for x in (n_budget, max_requests) if x)
    todo = pending(images_dir, raw_dir, out, limit=limit,
                   shards=shards, shard_index=shard_index)
    if not todo:
        where = f" for shard {shard_index}/{shards}" if shards > 1 else ""
        print(f"[submit] nothing pending{where} — everything is done or in flight.")
        return
    cap = "budget" if n_budget and n_budget <= max_requests else "request cap"
    print(f"[submit] budget ${max_spend:g} at ~${per_det*1000:.3f}/1k dets "
          f"=> {n_budget:,} dets; job cap {max_requests:,}; "
          f"{len(todo):,} selected (limited by {cap})")

    reqs, keys, meta_rows = [], [], []
    cur_stem, img = None, None
    for p, stem, i in todo:
        if stem != cur_stem:
            if img:
                img.close()
            img = Image.open(p).convert("RGB")
            cur_stem = stem
        rec = load_raw(raw_dir, stem)
        det = rec["detections"][i]
        b64s, crop = _crop_b64(img, det)
        key = f"{stem}::{i}"
        reqs.append({"key": key, "request": {
            "contents": [{"role": "user", "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64s}}]}],
            "generation_config": {"max_output_tokens": max_tokens}}})
        keys.append((stem, i))
        meta_rows.append({"image_stem": stem, "det_index": i, "image": p.name,
                          "img_wh": [rec.get("width", img.width),
                                     rec.get("height", img.height)],
                          "sam_class_id": det["cls"], "sam_conf": det.get("conf"),
                          "box": det["box"], "n_parts": len(det.get("parts", [])),
                          "blurriness": round(blurriness(img, det["box"]) or 0, 1),
                          "person_px_height": round(det["box"][3] - det["box"][1], 1)})
    if img:
        img.close()

    # Shard suffix keeps concurrent coordinators from colliding: two pods that
    # finish building in the same second would otherwise write the same
    # requests_/pending_/meta_ filenames and one would clobber the other's
    # reservation, silently freeing those detections for a double submit.
    short = time.strftime("%Y%m%d-%H%M%S") + (f"-s{shard_index}" if shards > 1 else "")
    est = len(reqs) * per_det

    if dry_run:
        # Inspection only: write the payload for eyeballing but do NOT write
        # the pending_/meta_ reservation files. Writing them would mark these
        # detections in flight with no job behind them, blocking them from a
        # later real submit until the orphan was deleted by hand.
        rf = _bdir(out) / f"dryrun_{short}.jsonl"
        rf.write_text("\n".join(json.dumps(r) for r in reqs) + "\n")
        print(f"[submit] wrote {len(reqs):,} requests -> {rf} "
              f"(~{rf.stat().st_size/1e6:.1f} MB, est ${est:.2f})")
        print("[submit] --dry-run: nothing uploaded, nothing reserved. "
              "Inspect the file, then re-run without --dry-run "
              "(the same detections will be selected).")
        return

    rf = _bdir(out) / f"requests_{short}.jsonl"
    rf.write_text("\n".join(json.dumps(r) for r in reqs) + "\n")
    (_bdir(out) / f"pending_{short}.json").write_text(json.dumps(keys))
    (_bdir(out) / f"meta_{short}.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta_rows) + "\n")
    print(f"[submit] wrote {len(reqs):,} requests -> {rf} "
          f"(~{rf.stat().st_size/1e6:.1f} MB, est ${est:.2f})")

    from google import genai
    client = genai.Client()
    try:
        up = client.files.upload(file=str(rf),
                                 config={"display_name": f"spotlight_{short}",
                                         "mime_type": "application/jsonl"})
        job = client.batches.create(model=model, src=up.name,
                                    config={"display_name": f"spotlight_{short}"})
    except Exception:
        # The reservation was written before the upload; a failure here (the
        # 20 GiB file_storage_bytes quota 429 is the observed one) would leave
        # a phantom in-flight reservation that permanently blocks these
        # detections from reselection. One night of this leaked 650k
        # detections. Roll the reservation back so a failed submit costs
        # nothing but a retry.
        (_bdir(out) / f"pending_{short}.json").unlink(missing_ok=True)
        (_bdir(out) / f"meta_{short}.jsonl").unlink(missing_ok=True)
        rf.unlink(missing_ok=True)
        raise
    jobs = _jobs(out)
    jobs.append({"short_id": short, "job_name": job.name, "model": model,
                 "n_requests": len(reqs), "est_cost_usd": round(est, 4),
                 "submitted_at": short, "state": str(job.state),
                 "collected": False, "prompt_version": PROMPT_VERSION})
    _save_jobs(out, jobs)
    print(f"[submit] job {job.name} state={job.state}\n"
          f"[submit] poll with:  python3 spotlight_batch.py poll --out {out}")


def poll(out: Path):
    from google import genai
    client = genai.Client()
    jobs = _jobs(out)
    for j in jobs:
        if j.get("collected"):
            print(f"  {j['short_id']}  COLLECTED  ({j['n_requests']} reqs)")
            continue
        job = client.batches.get(name=j["job_name"])
        j["state"] = str(job.state)
        print(f"  {j['short_id']}  {j['state']}  ({j['n_requests']} reqs, "
              f"est ${j['est_cost_usd']:.2f})")
    _save_jobs(out, jobs)
    return jobs


def _result_lines(client, job):
    """Yield parsed result rows from a finished job's output file."""
    dest = getattr(job, "dest", None)
    name = getattr(dest, "file_name", None) or getattr(job, "output_file", None)
    if not name:
        raise SystemExit(f"job {job.name} has no output file; state={job.state}")
    raw = client.files.download(file=name)
    text = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def collect(out: Path, rate_in=BATCH_RATE_IN, rate_out=BATCH_RATE_OUT):
    from google import genai
    client = genai.Client()
    jobs = _jobs(out)
    vpath = out / "verdicts_batch.jsonl"
    total_new = 0
    for j in jobs:
        if j.get("collected"):
            continue
        job = client.batches.get(name=j["job_name"])
        state = str(job.state)
        if "SUCCEEDED" not in state and "COMPLETED" not in state:
            print(f"  {j['short_id']}: {state} — not ready")
            continue
        meta = {}
        mf = _bdir(out) / f"meta_{j['short_id']}.jsonl"
        for line in mf.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                meta[f"{m['image_stem']}::{m['det_index']}"] = m
        n, tin, tout = 0, 0, 0
        with vpath.open("a") as fh:
            for row in _result_lines(client, job):
                key = row.get("key")
                m = meta.get(key)
                if m is None:
                    continue
                resp = row.get("response") or {}
                text = ""
                try:
                    text = resp["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    text = json.dumps(resp)[:400] if resp else ""
                usage = resp.get("usageMetadata") or resp.get("usage_metadata") or {}
                tin += usage.get("promptTokenCount", 0) or 0
                tout += ((usage.get("candidatesTokenCount", 0) or 0)
                         + (usage.get("thoughtsTokenCount", 0) or 0))
                v = parse_verdict(text)
                fh.write(json.dumps(dict(m, prompt_sha=j.get("prompt_sha", ""),
                                         parse_ok=v is not None, v=v,
                                         raw_text=text, source="batch")) + "\n")
                n += 1
        cost = (tin * rate_in + tout * rate_out) / 1e6
        j.update(collected=True, n_collected=n, input_tokens=tin,
                 output_tokens=tout, actual_cost_usd=round(cost, 4))
        total_new += n
        print(f"  {j['short_id']}: collected {n:,} verdicts · "
              f"{tin:,} in / {tout:,} out tokens · ACTUAL ${cost:.3f} "
              f"(estimated ${j['est_cost_usd']:.3f})")
    _save_jobs(out, jobs)
    print(f"[collect] {total_new:,} new verdicts -> {vpath}")


def status(images_dir: Path, raw_dir: Path, out: Path):
    done = done_keys(out)
    infl = inflight_keys(out)
    total = 0
    for p in sorted(q for q in images_dir.rglob("*") if q.suffix.lower() in IMG_EXTS):
        rec = load_raw(raw_dir, p.stem)
        if rec:
            total += len(rec.get("detections", []))
    jobs = _jobs(out)
    spent = sum(j.get("actual_cost_usd") or 0 for j in jobs)
    est_inflight = sum(j["est_cost_usd"] for j in jobs if not j.get("collected"))
    print(json.dumps({
        "detections_total": total,
        "verdicts_done": len(done),
        "in_flight": len(infl),
        "pending": total - len(done) - len(infl),
        "pct_done": round(100 * len(done) / total, 2) if total else 0,
        "jobs": len(jobs),
        "actual_spend_usd": round(spent, 4),
        "in_flight_est_usd": round(est_inflight, 4),
        "projected_total_usd": round(
            (spent + est_inflight) / max(1e-9, (len(done) + len(infl))) * total, 2)
        if (len(done) + len(infl)) else None}, indent=2))


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        (r / "img").mkdir(); (r / "raw").mkdir(); (r / "run").mkdir()
        for name in ("a", "b"):
            Image.new("RGB", (200, 200), (100,) * 3).save(r / f"img/{name}.jpg")
            (r / f"raw/{name}.txt").write_text("1 0.1 0.1 0.4 0.1 0.4 0.9\n"
                                               "0 0.5 0.5 0.9 0.5 0.9 0.9\n")
            json.dump({"image": f"{name}.jpg", "width": 200, "height": 200,
                       "detections": [
                           {"cls": 1, "conf": 0.9, "box": [20, 20, 80, 180],
                            "parts": [[[20, 20], [80, 20], [80, 180]]]},
                           {"cls": 0, "conf": 0.8, "box": [100, 100, 180, 180],
                            "parts": []}]}, open(r / f"raw/{name}.json", "w"))

        # 4 detections pending
        assert len(pending(r / "img", r / "raw", r / "run")) == 4

        # budget selection: $ that buys exactly 2
        per = est_cost_per_det()
        submit(r / "img", r / "raw", r / "run", per * 2.4, "m", 400, dry_run=True)
        dry = list((r / "run/batch").glob("dryrun_*.jsonl"))
        assert len(dry) == 1 and sum(1 for _ in dry[0].open()) == 2, dry
        # a DRY RUN MUST NOT RESERVE: no pending file, nothing in flight,
        # and the same work still selectable by a real submit
        assert not list((r / "run/batch").glob("pending_*.json")), "dry run reserved!"
        assert inflight_keys(r / "run") == set()
        assert len(pending(r / "img", r / "raw", r / "run")) == 4

        # simulate a REAL submit reserving 2 -> in flight, no double-pay
        (r / "run/batch/pending_realjob.json").write_text(
            json.dumps([["a", 0], ["a", 1]]))
        jobs = [{"short_id": "realjob",
                 "job_name": "x", "n_requests": 2, "est_cost_usd": 0.0,
                 "collected": False, "state": "PENDING"}]
        _save_jobs(r / "run", jobs)
        assert len(inflight_keys(r / "run")) == 2
        assert len(pending(r / "img", r / "raw", r / "run")) == 2

        # a live verdict for one of them also removes it from pending
        (r / "run/verdicts.jsonl").write_text(json.dumps(
            {"image_stem": "a", "det_index": 0}) + "\n")
        assert ("a", 0) in done_keys(r / "run")

        status(r / "img", r / "raw", r / "run")
    print("\nspotlight_batch.py self-test passed")


def main():
    ap = argparse.ArgumentParser(description="Spotlight Batch API path")
    ap.add_argument("cmd", nargs="?",
                    choices=["submit", "poll", "collect", "status"])
    ap.add_argument("--images"); ap.add_argument("--raw-labels"); ap.add_argument("--out")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-requests", type=int, default=MAX_REQUESTS_PER_JOB,
                    help=f"hard cap on requests per job (default "
                         f"{MAX_REQUESTS_PER_JOB:,}; each carries a ~40 KB crop)")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the image list N ways so several machines can "
                         "build crops in parallel (crop building is ~35 ms/det "
                         "and is the pipeline's throughput ceiling)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="which slice this machine handles, 0..--shards-1")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the request file but do not upload or spend")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    if a.cmd == "submit":
        if not (a.images and a.raw_labels and a.max_spend):
            ap.error("submit needs --images --raw-labels --max-spend")
        if not (0 <= a.shard_index < a.shards):
            ap.error("--shard-index must be in [0, --shards)")
        lock_name = f"_s{a.shard_index}" if a.shards > 1 else ""
        with SubmitLock(out, name=lock_name):
            submit(Path(a.images), Path(a.raw_labels), out, a.max_spend,
                   a.model, a.max_tokens, a.dry_run, a.max_requests,
                   a.shards, a.shard_index)
    elif a.cmd == "poll":
        poll(out)
    elif a.cmd == "collect":
        collect(out)
    elif a.cmd == "status":
        if not (a.images and a.raw_labels):
            ap.error("status needs --images --raw-labels")
        status(Path(a.images), Path(a.raw_labels), out)
    else:
        ap.error("cmd must be submit / poll / collect / status")


if __name__ == "__main__":
    main()
