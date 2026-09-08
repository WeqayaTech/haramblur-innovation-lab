# Pipeline v1 pilot — accuracy & pricing (paste into ClickUp)

> Paste each block below into a ClickUp doc — the `|` markdown converts to a table
> automatically. If pasting into a comment instead, use /table and copy the cells.

Pipeline: SAM3 detects every person → each detection sent to Gemini Flash-Lite as a padded
crop with the person's mask outlined → Lite confirms real person + corrects gender/age →
age > 12 defaults to adult, rejected detections deleted.

## Accuracy (pilot — 25 images per dataset)

| What we measured | Result |
| --- | --- |
| Real people kept — regular photos | 26/26 — **100%** |
| Real people kept — crowd scenes | 312/388 — 80% → **98% after fix** (confirmation run pending) |
| Gender correctness (adults) | 18/18 — **100%** |
| Overall label accuracy vs human ground truth | 23/26 — **88%** (all 3 misses: teenagers read as children) |
| False detections removed — empty scenes | 40/52 — **77%** |
| False detections removed — crowd scenes | 5/8 — **63%** |

## Pricing

| Item | Cost |
| --- | --- |
| Everything spent so far (pilots, fix, traces) | **~$0.60** |
| Confirmation run | ~$0.40 |
| Full experiment (1,500 images ≈ 13,800 detections) | **~$8** |
| Production — 500,000 ordinary photos | **~$380** (Batch) |
| Production — 500,000 crowd-heavy images | ~$3,190 (Batch) |

Unit price: **~$0.61 per 1,000 people labeled** — cost scales with people, not images.
Dollar figures pending verification against Google's official rate card (token counts are
measured).

Attachments for the full story: `trace-lagenda.html` / `trace-crowd.html` /
`trace-pass.html` (stage-by-stage walkthroughs) · `failed_recovery_report.html`
(the crowd fix, before/after) · `EXP-2026-10-TEAM-BRIEF.md` (full brief).
