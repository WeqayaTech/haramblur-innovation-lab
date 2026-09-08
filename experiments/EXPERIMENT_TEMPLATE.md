# Experiment: [plain question — e.g. "Can X do Y well enough to use?"]

**In one line:** _[one sentence — what you tested and why it matters. Keep it Short.]_

**Status:** _[Planned / Running / Done]_ · _[sample size]_ · Owner: _[name]_

> **How to use this template.** Write **"What we wanted to find out"** — including the "good enough" bar —
> **before you run anything.** Deciding the bar before you see the numbers is the one habit that keeps an
> experiment honest. Fill in results and decisions after. Write the whole thing in plain terms, like you're
> explaining it to a teammate — define any extra details in the appendix, not in the body. Delete these italic notes
> as you go.

---

## The short version

_3–6 bullets a busy teammate can read and walk away understanding. Lead with the answer, not the setup.
Include the main win, the main weakness, and the biggest caveat. Write this section LAST, once you know the
story._

- ...

## Why we did this

_2–4 sentences: the real-world problem and what decision this will inform. Plain terms._

## What we wanted to find out

_The question in one plain sentence._

_Then — before running — write down what "good enough" means, as specific numbers:_

- **[thing]:** _right at least [X]% of the time_
- **[thing]:** _[bar]_

## How we did it

_Numbered, plain steps anyone could follow:_

1. **The answer key.** _What's your ground truth, and why do you trust it?_
2. **What we did.** _What you actually ran / asked, in plain terms._
3. **How we compared.** _How you scored it against the answer key._
4. **Definitions that matter.** _Any cutoff or term the result depends on (e.g. where "child" ends)._

_If a design choice isn't obvious, say why in one line — future readers will wonder. (Example: "we asked it
blind so it couldn't just agree with the label.")_

## What we found

_The results in plain terms, with a chart per key point — clearest chart first. For each finding: the number,
what it means, and a picture if it helps. Note which mistakes are the "safe" ones and which are dangerous for
us._

- **[headline result]:** _..._

  ![caption](assets/png/chart-name.png)

## What we can decide from this

_Turn each finding into an action or a decision. Be concrete: "turn X on, with a human handling the unsure
cases", "don't use it for Y", "we still need to decide Z". If you can, translate the result into the number
leadership cares about — cost or effort saved (e.g. "cuts manual review ~80%") — not just accuracy._

## What this does NOT tell us

_Be the harshest reader of your own result. What would make a good score misleading? What did the test not
cover? (Example: "tested on general data, not the specific hard cases where we know we struggle.") List the
honest gaps — this is what stops people over-claiming._

## What's next

_The follow-up experiments or steps, named._

---

## Appendix — the details (for anyone who wants them)

_Everything needed to reproduce or scrutinize, kept out of the main read:_

- **Exact method / prompt / config** used, frozen — note if anything changed mid-run (it shouldn't).
- **Sample details, scripts, and the command(s)** to re-run.
- **Plain-word definitions** of any metric you used. Define the jargon here so the body stays simple.
  _(Example: "'not sure / coverage' = how often it declines to answer instead of committing.")_

### Appendix — "verify it yourself" (MANDATORY, write BEFORE running)

_Reviewers don't trust a prose summary of what the code did — they want to check the mechanism.
Paste **verbatim code snippets** (not paraphrases) with `file:line` references, re-reading the
source first so they're current. Cover every item below that applies, and **say honestly what was
NOT done** (e.g. a threshold picked by intuition, not calibrated). See the
`reproducibility-appendix-convention` memory._

- **How the model was called** — exact model id/weights, the full prompt or text prompts,
  decoding params (VLM/LLM); or the exact invocation + settings for a local model (YOLO/SAM3).
- **How objects were compared / matched** — the actual matching function, quoted in full, plus
  every threshold it uses. (This is where the box-overlap bug hid — don't summarize it away.)
- **How samples were selected** — the selection/loading code and any random seed.
- **How balanced the data is** — the class / age / gender distribution of the evaluation set.
- **Every threshold & configuration** — a table of each magic number and the source line it lives
  on (match IoU, model conf, NMS IoU, age cutoffs, crowd/near-miss thresholds, …).

### Turning charts into images for docs (ClickUp, etc.)

_If your charts are SVGs, convert them to PNG first — many doc tools (ClickUp) don't render SVG. On a Mac with
Chrome installed, `experiments/make_charts.py` shows the pattern, and Chrome headless converts SVG→PNG at 2×.
Then paste the write-up text and drag the PNGs in._
