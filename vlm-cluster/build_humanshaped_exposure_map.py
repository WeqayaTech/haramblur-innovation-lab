import os, sys
from pathlib import Path

MODELS = ["y26n_humanshaped_v2", "y26s_humanshaped_smallpatch_v1"]
TIERS = ["t0_covered", "t1_modest", "t2_ordinary", "t3_revealing", "t4_high"]
SUBSET_ROOT = Path("/workspace/deploycmp/subsets")
HOLDOUT_EVAL = Path("/workspace/holdout_eval")

for model in MODELS:
    raw_src = HOLDOUT_EVAL / model / "raw"
    assert raw_src.exists(), raw_src
    for tier in TIERS:
        tdir = SUBSET_ROOT / f"w_exposure_{tier}"
        img_dir = tdir / "images"
        stems = [p.stem for p in img_dir.iterdir()]
        out_dir = tdir / f"raw_{model}"
        out_dir.mkdir(exist_ok=True)
        made = 0
        missing = 0
        for stem in stems:
            src = raw_src / f"{stem}.json"
            dst = out_dir / f"{stem}.json"
            if dst.exists() or dst.is_symlink():
                continue
            if not src.exists():
                missing += 1
                continue
            dst.symlink_to(src)
            made += 1
        print(f"{model} w_exposure_{tier}: {len(stems)} stems, {made} linked, {missing} missing")
