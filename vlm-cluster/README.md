# vlm-cluster — HaramBlur evaluation and labeling tools

All scripts live here. **Before writing a new one, check `docs/CODEMAP.md`** — it has one
row per script with purpose, status, and import dependencies.

## What's in here

- **Frozen core** (`run_model_children.py`, `translation.py`, `two_axis.py`, `spotlight_run.py`)
  — imported by many other scripts; treat signature changes as breaking.
- **Labeling pipeline** — SAM3 detection (`autolabel_sam_raw.py`), Gemini verification
  (`spotlight_run.py`, `spotlight_batch.py`), label emission (`parallel_emit.py`).
- **Scoring / evaluation** — `eval_negatives_crowd.py`, `run_autolabel_on_manifest.py`,
  `map_eval.py`, `conf_sweep.py`, `pr_curve.py`, `ap_sweep.py`, `deploy_compare.py`.
- **Training** — `train_gradsuppress.py`, `train_small_object_patches.py`.
- **Export / deployment** — `float_head_quant.py`, `bench_matrix.py`, `tfjs_build_two.sh`,
  the `matrix_lib.sh` / `run_cells.sh` multi-pod harness.
- **Galleries and reports** — `build_error_gallery.py`, `deploy_report.py`,
  `women_report_final.py`, and others.
- **Historical** — scripts that ran once for a closed experiment, kept for reference.
  See CODEMAP's status column.

## Quick start (no GPU, no data)

Most eval tools self-test synthetically:

```bash
python3 translation.py
python3 eval_negatives_crowd.py --selftest
python3 map_eval.py --selftest
python3 conf_sweep.py --selftest
```

## Pod setup

See `SETUP_BLACKWELL.md` for Blackwell (sm_120) pods. For L4/A100 pods, the default
torch works; install missing deps with `pod_setup.sh`.
