#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-06: turn measured token counts into verified dollars.

Walks every run dir under --root, re-multiplies the saved token counts
(cost_report.json — measured from vendor API responses, never estimated) by
the rates in model_pricing.json, and rewrites cost_usd + the summary's
labeling_economics dollars. Run it after hand-filling model_pricing.json from
the vendors' own pricing pages; re-run any time a rate changes. No API calls.

    python3 recompute_costs.py --root /workspace/exp06
    python3 recompute_costs.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRICING_PATH = Path(__file__).parent / "model_pricing.json"


def recompute(root: Path, pricing: dict) -> list:
    rows = []
    for cr_path in sorted(root.glob("*/*/cost_report.json")):
        cr = json.loads(cr_path.read_text())
        model = cr.get("model_id")
        rate = pricing.get(model, {})
        rin, rout = rate.get("input_per_1m"), rate.get("output_per_1m")
        if rin is None or rout is None:
            rows.append((str(cr_path.parent.relative_to(root)), model,
                         None, "rate still null in model_pricing.json"))
            continue
        usd = (cr["input_tokens"] / 1e6 * rin
               + cr["output_tokens"] / 1e6 * rout)
        cr["cost_usd"] = round(usd, 6)
        cr["cost_usd_per_1000_calls"] = (round(usd / cr["calls"] * 1000, 4)
                                         if cr.get("calls") else None)
        cr["rates_used"] = {"input_per_1m": rin, "output_per_1m": rout}
        cr_path.write_text(json.dumps(cr, indent=2))

        sm_path = cr_path.parent / "summary.json"
        if sm_path.exists():
            sm = json.loads(sm_path.read_text())
            econ = sm.get("labeling_economics")
            if econ:
                n_img = econ.get("images") or 0
                n_per = econ.get("persons_labeled") or 0
                econ["usd_per_1000_images"] = (round(usd / n_img * 1000, 2)
                                               if n_img else None)
                econ["usd_per_1000_persons"] = (round(usd / n_per * 1000, 2)
                                                if n_per else None)
                sm_path.write_text(json.dumps(sm, indent=2))
        rows.append((str(cr_path.parent.relative_to(root)), model,
                     round(usd, 4), ""))
    return rows


def main():
    ap = argparse.ArgumentParser(description="recompute $ from measured tokens")
    ap.add_argument("--root", default="/workspace/exp06")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    pricing = json.loads(PRICING_PATH.read_text())
    rows = recompute(Path(args.root), pricing)
    total = 0.0
    for run, model, usd, note in rows:
        print(f"  {run:<30} {model:<26} "
              f"{('$' + format(usd, '.4f')) if usd is not None else '—':>10}  {note}")
        total += usd or 0.0
    print(f"\n[recompute] total across runs with verified rates: ${total:.4f}")
    print("[recompute] cross-check this total against each vendor's billing "
          "dashboard — if they agree, the rates AND the token accounting are "
          "both verified end-to-end.")


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "m" / "ds"
        run.mkdir(parents=True)
        (run / "cost_report.json").write_text(json.dumps(
            {"model_id": "test-model", "calls": 3,
             "input_tokens": 2_000_000, "output_tokens": 1_000_000,
             "cost_usd": None}))
        (run / "summary.json").write_text(json.dumps(
            {"labeling_economics": {"images": 4, "persons_labeled": 10,
                                    "usd_per_1000_images": None,
                                    "usd_per_1000_persons": None}}))
        rows = recompute(Path(td), {"test-model": {"input_per_1m": 2.0,
                                                    "output_per_1m": 10.0}})
        assert rows[0][2] == 14.0, rows          # 2M*2 + 1M*10 per 1M
        sm = json.loads((run / "summary.json").read_text())
        assert sm["labeling_economics"]["usd_per_1000_images"] == 3500.0
        assert sm["labeling_economics"]["usd_per_1000_persons"] == 1400.0
        rows2 = recompute(Path(td), {})          # missing rate -> flagged, not guessed
        assert rows2[0][2] is None
    print("[recompute_costs] selftest OK")


if __name__ == "__main__":
    main()
