#!/usr/bin/env python3
"""
HARAMBLUR VLM-cluster — Stage 1: describe every detected person.

For each image we generate ONE scene caption (shared by all its boxes), then for
each Woman/Man/Child box a structured JSON description from a tight padded crop.
The YOLO class is fed as a hint so the model anchors instead of guessing.

Output: descriptions.jsonl (one record per object) + crops/<id>.jpg thumbnails.
Resumable — re-running skips ids already in the jsonl.

Engine: local Qwen2.5-VL-7B-Instruct (default). Use --mock to validate the
pipeline with synthetic descriptions and no GPU/model.

    python describe.py --yaml /workspace/open-images-v7/dataset.yaml \
        --split train --max-crops 1500 --out ./run1

    python describe.py --yaml ... --mock --max-crops 200 --out ./smoke   # no GPU
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import zlib
from pathlib import Path

import cv2

import dataset_utils as du

try:  # pulls OPENAI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------
# Description schema — structured fields cluster far better than free prose.
# "unknown" is always allowed; the model is told to only report what it sees.
# --------------------------------------------------------------------------

OBJECT_SCHEMA = {
    "apparent_age_band": "infant|toddler|child|preteen|teenager|young_adult|adult|middle_aged|elderly|senior|unknown",
    "apparent_gender": "man|woman|unknown",
    "age_read_confidence": "clear|ambiguous",
    "gender_read_confidence": "clear|ambiguous",
    "gender_cues": "short free-text: which cues drove the read (beard/clothing/hair/body/face)",
    "build": "slim|average|heavy|unknown",
    "facial_hair": "none|stubble|moustache|short_beard|full_beard|unknown",
    "hair_length": "none_or_covered|short|medium|long|unknown",
    "head_covering": "none|cap_hat|hijab|ghutra_keffiyeh|turban|helmet|other|unknown",
    "garment_type": "thobe_robe|abaya|dress|shirt_trousers|suit|sportswear|swimwear|other|unknown",
    "apparent_attire_region": "gulf_arab|south_asian|african|east_asian|western|other|unknown",
    "clothing": "short free-text, e.g. 'red t-shirt and jeans'",
    "clothing_coverage": "fully_covered|modest|moderate|revealing|minimal|unknown",
    "pose": "standing|sitting|walking|running|lying|other|unknown",
    "orientation": "frontal|three_quarter|profile|back|unknown",
    "occlusion": "none|partial|heavy",
    "visible_part": "face_only|upper_body|full_body|body_part|unknown",
    "activity": "short free-text, e.g. 'playing football'",
    "distinctive_features": "short free-text or 'none'",
    "skin_tone_mst": "Monk Skin Tone 1-10 (1 lightest .. 10 darkest) or unknown",
    "skin_tone_confidence": "reliable|uncertain_lighting",
}

SCENE_SCHEMA = {
    "setting": "indoor|outdoor|unknown",
    "scene_type": "street|home|beach|pool|sports_field|stage|office|nature|vehicle|other|unknown",
    "crowd": "solo|small_group|crowd",
    "short_caption": "one neutral sentence describing the whole scene",
}

OBJECT_PROMPT = (
    "You are annotating a person-detection dataset. A detector labeled this crop "
    "'{cls}', but it can be WRONG. Describe ONLY what is clearly visible; use "
    "'unknown' when unsure. Do not identify individuals.\n"
    "Guidance:\n"
    "- apparent_age_band: ALWAYS pick the single closest band as your best "
    "estimate, even if you are not certain of the exact band — infant (<1), "
    "toddler (1-3), child (4-9), preteen (10-12, pre-puberty), teenager (13-17), "
    "young_adult (18-29), adult (30-49), middle_aged (50-64), elderly (65-79), "
    "senior (80+).\n"
    "- apparent_gender: ALWAYS pick man or woman as your best estimate from "
    "visible cues, even if not fully certain; use 'unknown' ONLY when no "
    "gender cue is visible (fully covered figure, back turned, no face).\n"
    "- age_read_confidence: 'clear' if the broad life stage (child / teenager / "
    "adult / elderly) is evident; 'ambiguous' ONLY when you genuinely cannot "
    "tell the broad stage. Do NOT mark ambiguous merely because the exact band "
    "is uncertain.\n"
    "- gender_read_confidence: 'clear' if apparent gender is evident; "
    "'ambiguous' ONLY when it truly cannot be determined from what is visible "
    "(e.g. a fully robed figure with no visible face, or a beardless/feminine "
    "person that genuinely reads either way). Judge gender and age from visible "
    "cues, NOT from the '{cls}' label, which can be wrong. Record the cues you "
    "used in gender_cues.\n"
    "- facial_hair: be specific (none/stubble/moustache/short_beard/full_beard).\n"
    "- head_covering / garment_type: name the actual garment "
    "(e.g. thobe_robe, abaya, ghutra_keffiyeh, hijab).\n"
    "- apparent_attire_region: a ROUGH guess of clothing STYLE only, never "
    "nationality or ethnicity; use 'unknown' if unclear.\n"
    "- skin_tone_mst: pick the closest Monk Skin Tone number, 1 (lightest) to "
    "10 (darkest), from facial skin. If lighting is shadowed or over/under-"
    "exposed, still estimate but set skin_tone_confidence='uncertain_lighting'.\n"
    "Return a single JSON object with EXACTLY these keys:\n{schema}\n"
    "Return only the JSON."
)

SCENE_PROMPT = (
    "Describe the overall scene of this image neutrally and factually. Return a "
    "single JSON object with EXACTLY these keys:\n{schema}\nReturn only the JSON."
)


def schema_str(schema: dict) -> str:
    return "\n".join(f'  "{k}": ({v})' for k, v in schema.items())


def extract_json(text: str) -> dict:
    """Pull the model's JSON answer out of its output.

    Fast path: the whole first-{ .. last-} span parses as one object. When it
    doesn't (thinking models sometimes emit a JSON block, then prose like
    "Wait, I need to rescale...", then a corrected block — seen live from
    qwen3.7-plus), fall back to scanning for every balanced JSON object and
    return the LAST one: the model's final, self-corrected answer.
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    # common fixups: trailing commas; orphan keys (a key immediately followed
    # by another key-colon is always invalid JSON — seen live from
    # gemini-3.6-flash: {"box_2d": [...], "label": "gender": "man", ...})
    cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    cleaned = re.sub(r'"[A-Za-z_]+":\s*(?="[A-Za-z_]+":)', "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # multi-block fallback: collect every balanced object, keep the last
    dec = json.JSONDecoder()
    objs, i = [], 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict):
                objs.append(obj)
            i = max(end, j + 1)
        except json.JSONDecodeError:
            i = j + 1
    return objs[-1] if objs else {}


def coerce(raw: dict, schema: dict) -> dict:
    """Keep only schema keys; fill missing with 'unknown'."""
    out = {}
    for k in schema:
        v = raw.get(k, "unknown")
        out[k] = str(v).strip() if v not in (None, "") else "unknown"
    return out


# --------------------------------------------------------------------------
# Describers
# --------------------------------------------------------------------------

class MockDescriber:
    """Synthetic descriptions so the whole pipeline runs without a GPU."""

    def __init__(self, seed=0):
        self.r = random.Random(seed)

    def _pick(self, spec):
        opts = [o for o in spec.split("|") if "free-text" not in o
                and "sentence" not in o and "e.g." not in o]
        return self.r.choice(opts) if opts else "sample"

    def scene(self, image_bgr):
        return coerce({k: self._pick(v) for k, v in SCENE_SCHEMA.items()}, SCENE_SCHEMA)

    def describe(self, crop_bgr, cls, scene=None, schema=None):
        sch = schema or OBJECT_SCHEMA
        d = {k: self._pick(v) for k, v in sch.items()}
        if "apparent_age_band" in sch:
            d["apparent_age_band"] = ("child" if cls == "Child"
                                      else self.r.choice(["young_adult", "adult"]))
        if "apparent_gender" in sch:
            d["apparent_gender"] = ("woman" if cls == "Woman"
                                    else "man" if cls == "Man"
                                    else self.r.choice(["woman", "man", "unclear"]))
        return coerce(d, sch)

    def scene_batch(self, images, batch_size=1):
        return [self.scene(im) for im in images]

    def describe_batch(self, items, batch_size=1, prompt_template=None, schema=None):
        return [self.describe(c, cls, None, schema) for c, cls in items]


class QwenDescriber:
    """Local Qwen2.5-VL-7B-Instruct describer."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device=None,
                 max_new_tokens=320, max_pixels=1003520):
        import torch
        from transformers import (AutoProcessor,
                                   Qwen2_5_VLForConditionalGeneration)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=self.device)
        self.model.eval()
        # max_pixels caps tokens per image -> bounds VRAM + speeds things up
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        # left padding is required for correct batched decoder-only generation
        self.processor.tokenizer.padding_side = "left"
        from qwen_vl_utils import process_vision_info
        self._proc_vision = process_vision_info

    def _run_batch(self, items):
        """items: list[(image_bgr, prompt)] -> list[str] (one decode per item)."""
        from PIL import Image
        messages_list = []
        for image_bgr, prompt in items:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            messages_list.append([{"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(rgb)},
                {"type": "text", "text": prompt},
            ]}])
        texts = [self.processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True)
                 for m in messages_list]
        image_inputs, video_inputs = self._proc_vision(messages_list)
        inputs = self.processor(text=texts, images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]  # left-padded -> uniform offset
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

    def _chunked(self, items, prompts, schema, batch_size):
        out = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            raw = self._run_batch(chunk)
            out += [coerce(extract_json(o), schema) for o in raw]
        return out

    def scene_batch(self, images, batch_size=4):
        p = SCENE_PROMPT.format(schema=schema_str(SCENE_SCHEMA))
        return self._chunked([(im, p) for im in images], None,
                             SCENE_SCHEMA, batch_size)

    def describe_batch(self, items, batch_size=4, prompt_template=None, schema=None):
        tmpl = prompt_template or OBJECT_PROMPT
        sch_dict = schema or OBJECT_SCHEMA
        sch = schema_str(sch_dict)
        prepared = [(crop, tmpl.format(cls=cls, schema=sch))
                    for crop, cls in items]
        return self._chunked(prepared, None, sch_dict, batch_size)

    # single-item convenience (used by nothing hot; kept for ad-hoc use)
    def scene(self, image_bgr):
        return self.scene_batch([image_bgr], 1)[0]

    def describe(self, crop_bgr, cls, scene=None):
        return self.describe_batch([(crop_bgr, cls)], 1)[0]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_done(jsonl: Path) -> set:
    done = set()
    if jsonl.exists():
        with jsonl.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def fmt_dur(secs: float) -> str:
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def save_thumb(crop, path, thumb):
    s = thumb / max(crop.shape[:2])
    t = cv2.resize(crop, (max(1, int(crop.shape[1] * s)),
                          max(1, int(crop.shape[0] * s))))
    cv2.imwrite(str(path), t)


def main():
    ap = argparse.ArgumentParser(description="VLM-cluster stage 1: describe")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-crops", type=int, default=0, help="0 = all (no cap)")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--min-crop-px", type=int, default=24)
    ap.add_argument("--pad", type=float, default=0.25, help="context padding frac")
    ap.add_argument("--out", default="./run1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true", help="synthetic, no GPU/model")
    ap.add_argument("--engine", default="qwen",
                    choices=["qwen", "openai", "gemini", "claude"],
                    help="which describer to use (--mock overrides this)")
    ap.add_argument("--model", default=None,
                    help="model id; defaults per --engine "
                         "(Qwen/Qwen2.5-VL-7B-Instruct, gpt-5.6-sol, "
                         "gemini-3.5-flash, claude-sonnet-5)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="crops/scenes per GPU generate call (raise to max VRAM)")
    ap.add_argument("--chunk", type=int, default=64,
                    help="images processed + written per status tick")
    ap.add_argument("--max-pixels", type=int, default=1003520,
                    help="cap pixels per image (bounds VRAM); ~1.0MP default")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="generation length cap; lower = faster (JSON needs ~200)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total parallel processes (one per GPU)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="this process's shard index, 0..num_shards-1")
    args = ap.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard-id must be in [0, --num-shards)")

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (out / "descriptions.jsonl" if args.num_shards == 1
             else out / f"descriptions.shard{args.shard_id}.jsonl")
    done = load_done(jsonl)
    done_stems = {oid.rsplit("_", 1)[0] for oid in done}  # skip whole images fast
    if done:
        print(f"[describe] resuming — {len(done)} objects already described")

    cfg = du.load_yaml(Path(args.yaml))
    root = du.resolve_root(cfg, Path(args.yaml))
    names = du.class_names(cfg)
    img_dir, lbl_dir = du.discover_split(cfg, root, args.split)
    print(f"[describe] classes={list(names.values())}  images={img_dir}")

    if args.mock:
        engine = MockDescriber(seed=args.seed)
        print("[describe] MOCK engine (no GPU)")
    elif args.engine == "qwen":
        engine = QwenDescriber(model_id=args.model or "Qwen/Qwen2.5-VL-7B-Instruct",
                               device=args.device,
                               max_pixels=args.max_pixels,
                               max_new_tokens=args.max_new_tokens)
        print(f"[describe] Qwen2.5-VL on {engine.device} "
              f"(batch={args.batch_size}, max_pixels={args.max_pixels})")
    else:
        import api_describers
        engine = api_describers.build_engine(args.engine, args.model)
        print(f"[describe] API engine '{args.engine}' -> {engine.cost.model_id} "
              f"(sequential, batch-size arg ignored)")

    # --- build the work list once, skip done/other-shard BEFORE any decode ---
    imgs = list(du.list_images(img_dir))
    random.Random(args.seed).shuffle(imgs)
    if args.max_images:
        imgs = imgs[:args.max_images]

    def mine(p):
        if args.num_shards > 1 and (
                zlib.crc32(p.stem.encode()) % args.num_shards != args.shard_id):
            return False
        return p.stem not in done_stems

    pending = [p for p in imgs if mine(p)]
    total = len(pending)
    if args.num_shards > 1:
        print(f"[describe] shard {args.shard_id}/{args.num_shards} -> {jsonl.name}")
    print(f"[describe] {total} images to process this run "
          f"(of {len(imgs)} candidates)")
    if total == 0:
        print("[describe] nothing to do.")
        return

    n = 0           # new objects this run
    done_imgs = 0   # images processed this run
    t0 = time.time()
    f = jsonl.open("a")

    for start in range(0, total, args.chunk):
        chunk_paths = pending[start:start + args.chunk]
        entries = []        # (img_path, scene_idx, [(oid, crop, cls, box)])
        scene_imgs = []
        for img_path in chunk_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            raw = du.yolo_boxes(du.label_path_for(img_path, img_dir, lbl_dir), w, h)
            crops = []
            for bi, (cls_id, x1, y1, x2, y2) in enumerate(raw):
                oid = f"{img_path.stem}_{bi}"
                if oid in done:
                    continue
                nx1, ny1, nx2, ny2 = du.pad_box(x1, y1, x2, y2, w, h, args.pad)
                if nx2 - nx1 < args.min_crop_px or ny2 - ny1 < args.min_crop_px:
                    continue
                crop = img[ny1:ny2, nx1:nx2]
                if crop.size == 0:
                    continue
                crops.append((oid, crop, names.get(cls_id, str(cls_id)),
                              [int(x1), int(y1), int(x2), int(y2)]))
            if crops:
                scene_imgs.append(img)
                entries.append((img_path, len(scene_imgs) - 1, crops))

        if not entries:
            done_imgs += len(chunk_paths)
            continue

        scenes = engine.scene_batch(scene_imgs, args.batch_size)
        flat_objs = [(c, cls) for (_, _, crops) in entries
                     for (_, c, cls, _) in crops]
        obj_results = engine.describe_batch(flat_objs, args.batch_size)

        k = 0
        for (img_path, sidx, crops) in entries:
            scene = scenes[sidx]
            for (oid, crop, cls, box) in crops:
                obj = obj_results[k]
                k += 1
                save_thumb(crop, crops_dir / f"{oid}.jpg", args.thumb)
                f.write(json.dumps({
                    "id": oid, "image": str(img_path), "class": cls,
                    "box_xyxy": box, "scene": scene, "object": obj,
                }) + "\n")
                n += 1
        f.flush()
        done_imgs += len(chunk_paths)

        elapsed = time.time() - t0
        img_rate = done_imgs / elapsed if elapsed else 0
        crop_rate = n / elapsed if elapsed else 0
        if args.max_crops:  # ETA bound by the crop cap, not all images
            eta = (args.max_crops - n) / crop_rate if crop_rate else 0
            target = f"{n}/{args.max_crops} crops"
        else:
            eta = (total - done_imgs) / img_rate if img_rate else 0
            target = f"{done_imgs}/{total} imgs ({100*done_imgs/total:.1f}%)"
        print(f"[describe] {done_imgs}/{total} imgs · {n} crops "
              f"· {img_rate:.2f} img/s · {crop_rate:.2f} crop/s "
              f"· elapsed {fmt_dur(elapsed)} · ETA {fmt_dur(eta)} → {target}",
              flush=True)

        if args.max_crops and n >= args.max_crops:
            print(f"[describe] hit --max-crops {args.max_crops}, stopping")
            break

    f.close()
    print(f"[describe] done: {n} new objects this run "
          f"({done_imgs} imgs) -> {jsonl}")

    if hasattr(engine, "cost"):
        report = engine.cost.report()
        cost_path = out / "cost_report.json"
        with cost_path.open("w") as cf:
            json.dump(report, cf, indent=2)
        print(f"[describe] cost report -> {cost_path}: {report}")


if __name__ == "__main__":
    main()
