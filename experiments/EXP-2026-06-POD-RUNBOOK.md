# EXP-2026-06 — Pod runbook (you run, Claude reads outputs)

Same pattern as EXP-2026-02/03/04: copy-paste blocks; **→ paste back** marks output Claude
needs. **Any CPU pod works** — no Qwen run in this design, so no GPU needed. All four datasets
are already staged on the volume; there is no dataset phase.

**Never paste real API key values into chat or into any file that syncs back to the repo.**
Type them directly on the pod. Treat any key ever pasted into a chat as compromised — regenerate
it from the vendor console after this run.

---

## Phase 0 — copy code + verify the frozen pieces (minutes)

Copy these 5 files from local `vlm-cluster/` into the pod's
`/workspace/data_inspection_tools/vlm-cluster/` (scp, or paste via the web terminal):

- `vlm_detect_eval.py` (new — the whole run)
- `api_describers.py` (new)
- `model_pricing.json` (new)
- `describe.py` (changed — `extract_json` is imported from it; the pod copy predates the
  `--engine` edits, safest to sync it)
- `requirements.txt` (changed)

Then on the pod:

```bash
cd /workspace/data_inspection_tools/vlm-cluster
pip install -q openai google-genai anthropic python-dotenv

python3 vlm_detect_eval.py --selftest        # no network, no keys
python3 api_describers.py                    # ditto
# → paste back only if either fails

# datasets still where the registry says
ls /workspace/lagenda_eval/lagenda_yolo/                       # dataset.yaml gt.jsonl images labels
wc -l /workspace/lagenda_eval/lagenda_yolo/gt.jsonl            # 5000
ls /workspace/datasets/crowdhuman/ | head                      # Images  Images_sample500  annotation_val.odgt
find /workspace/datasets/pass_3k -type f | wc -l               # ~3000
find /workspace/datasets/object_set -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l   # ~259
# → paste back if anything looks off

mkdir -p /workspace/exp06
```

## Phase 1 — API keys (pod terminal, not chat)

```bash
cat > /workspace/data_inspection_tools/vlm-cluster/.env << 'EOF'
OPENAI_API_KEY=paste-real-key-here
GOOGLE_API_KEY=paste-real-key-here
ANTHROPIC_API_KEY=paste-real-key-here
DASHSCOPE_API_KEY=paste-real-key-here
EOF
```

Notes:
- The Google key must be a **Gemini Developer API / AI Studio** key (simple `api_key=` auth),
  not a Vertex service-account credential.
- The Qwen key is a **DashScope / Alibaba Cloud Model Studio** key
  (modelstudio.console.alibabacloud.com -> API keys; new accounts get a free token quota).
  Engine `qwenapi` talks to the DashScope international OpenAI-compatible endpoint, model
  `qwen3.7-plus` by default — the current flagship, natively multimodal (the older dedicated
  `qwen3-vl-*` line is a generation behind; `qwen3-vl-flash` remains a budget option). After
  its first crowd run,
  check Qwen's box convention with `reparse_boxes.py` diagnose mode before trusting matches —
  do NOT assume it honors the pixel x-first format (Gemini didn't).
- `model_pricing.json` rates are `null` on purpose — the smoke test runs fine without them
  (token counts + `cost_usd: null`). Fill them from each vendor's own pricing page only when
  you want a real $ figure.

## Phase 2 — the 16 runs (4 models × 4 datasets; ~48 API calls total, minutes)

```bash
cd /workspace/data_inspection_tools/vlm-cluster
set -a; source .env; set +a

LAG_IMG=/workspace/lagenda_eval/lagenda_yolo/images/val
LAG_LBL=/workspace/lagenda_eval/lagenda_yolo/labels/val
LAG_GT=/workspace/lagenda_eval/lagenda_yolo/gt.jsonl
CH_IMG=/workspace/datasets/crowdhuman/Images_sample500
CH_ODGT=/workspace/datasets/crowdhuman/annotation_val.odgt

for spec in "openai:gpt-5.6-sol:openai_sol" \
            "gemini:gemini-3.5-flash:gemini_flash" \
            "gemini:gemini-3.1-pro-preview:gemini_pro" \
            "claude:claude-sonnet-5:claude_sonnet5" \
            "qwenapi:qwen3.7-plus:qwen37_plus"; do
  eng="${spec%%:*}"; rest="${spec#*:}"; model="${rest%%:*}"; name="${rest#*:}"

  python3 vlm_detect_eval.py --engine "$eng" --model "$model" \
      --images "$LAG_IMG" --gt lagenda --labels "$LAG_LBL" --manifest "$LAG_GT" \
      --max-images 3 --seed 42 --out /workspace/exp06/$name/lagenda

  python3 vlm_detect_eval.py --engine "$eng" --model "$model" \
      --images "$CH_IMG" --gt crowdhuman --odgt "$CH_ODGT" \
      --max-images 3 --seed 42 --out /workspace/exp06/$name/crowd

  python3 vlm_detect_eval.py --engine "$eng" --model "$model" \
      --images /workspace/datasets/pass_3k --gt none \
      --max-images 3 --seed 42 --out /workspace/exp06/$name/pass

  python3 vlm_detect_eval.py --engine "$eng" --model "$model" \
      --images /workspace/datasets/object_set --gt none \
      --max-images 3 --seed 42 --out /workspace/exp06/$name/objects
done
```

If one engine's key isn't ready, run the loop body for the engines you have — every run is
independent. If a model id 404s (preview names get renamed), paste the error back; check the
vendor's live model list before assuming the harness is broken.

## Phase 3 — collect (seconds)

```bash
cd /workspace/exp06
for d in */*/; do
  echo "=== $d ==="
  python3 -c "
import json,sys
s=json.load(open('$d/summary.json'))
print({k:v for k,v in s.items() if k not in ('per_image','per_person')})
print(json.load(open('$d/cost_report.json')))"
done
# → paste back the whole output (16 blocks)
```

Also worth a skim: any `"parse_ok": false` or `"api_error": true` lines —

```bash
grep -l '"parse_ok": false\|"api_error": true' */*/detections.jsonl
# → paste back (ideally empty)
```

## What to paste back

1. Phase 0 dataset checks (only if something looks off).
2. The 16 summary+cost blocks from Phase 3.
3. The parse-failure grep result.

With those, Claude fills in the experiment doc's "What we found" and the scale-up
recommendation. Reminder: n=3 per dataset — plumbing + cost ballpark only, no accuracy verdict
(the doc's smoke-test caveat governs how these numbers get written up).

## After the run

- Regenerate any API key that was ever pasted into a chat.
- If ≥2 engines pass plumbing: the scale-up is the same loop with `--max-images 250` and real
  pre-registered bars added to the experiment doc first (see "What's next" in the doc).
