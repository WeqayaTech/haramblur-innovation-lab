"""
HARAMBLUR VLM-cluster — API-based describers for EXP-2026-06 (cost/accuracy
comparison of commercial VLMs against the local Qwen2.5-VL-7B baseline).

Same OBJECT_SCHEMA / OBJECT_PROMPT / SCENE_SCHEMA / SCENE_PROMPT and the same
extract_json/coerce parsing as describe.py's QwenDescriber, so the prompt is
held fixed across engines -- only the model changes. Each class tracks token
usage and, where model_pricing.json has a non-null rate, a running USD cost.

Requires per-provider SDKs only when that engine is actually selected:
    pip install openai google-genai anthropic

API keys read from the environment: OPENAI_API_KEY, GOOGLE_API_KEY,
ANTHROPIC_API_KEY (or GEMINI_API_KEY as a fallback for Google).
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2

# describe.py is imported LAZILY (inside the crop-description methods below)
# so this module works standalone in the production deployment, where only the
# raw _call path is used and describe.py's analysis stack is absent.

PRICING_PATH = Path(__file__).parent / "model_pricing.json"


def _load_pricing() -> dict:
    if not PRICING_PATH.exists():
        return {}
    with PRICING_PATH.open() as f:
        return json.load(f)


def _encode_jpeg(crop_bgr, quality=90) -> bytes:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


class CostTracker:
    """Shared usage/cost bookkeeping for every API describer."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.calls = 0
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_seconds = 0.0
        rate = _load_pricing().get(model_id, {})
        self.rate_in = rate.get("input_per_1m")
        self.rate_out = rate.get("output_per_1m")
        if self.rate_in is None or self.rate_out is None:
            print(f"[api_describers] WARNING: no verified $/1M-token rate for "
                  f"'{model_id}' in model_pricing.json -- cost will report as "
                  f"null, token counts only. Fill in the rate from the vendor's "
                  f"own pricing page before trusting a dollar figure.")

    def record(self, in_tok: int, out_tok: int, dt: float, error: bool = False):
        self.calls += 1
        self.wall_seconds += dt
        if error:
            self.errors += 1
            return
        self.input_tokens += in_tok
        self.output_tokens += out_tok

    @property
    def cost_usd(self):
        if self.rate_in is None or self.rate_out is None:
            return None
        return (self.input_tokens / 1_000_000 * self.rate_in
                + self.output_tokens / 1_000_000 * self.rate_out)

    def report(self) -> dict:
        return {
            "model_id": self.model_id,
            "calls": self.calls,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_seconds": round(self.wall_seconds, 1),
            "cost_usd": self.cost_usd,
            "cost_usd_per_1000_calls": (
                None if self.cost_usd is None or self.calls == 0
                else round(self.cost_usd / self.calls * 1000, 4)
            ),
        }


class _ApiDescriberBase:
    """Sequential (no batching) describer -- API latency dominates anyway."""

    def _call(self, image_bgr, prompt: str) -> tuple[str, int, int]:
        """Return (raw_text, input_tokens, output_tokens). Implement per vendor."""
        raise NotImplementedError

    def _run_one(self, image_bgr, prompt, schema):
        from describe import coerce, extract_json
        t0 = time.time()
        try:
            text, in_tok, out_tok = self._call(image_bgr, prompt)
            self.cost.record(in_tok, out_tok, time.time() - t0)
        except Exception as e:  # noqa: BLE001 -- log and degrade to "unknown", never crash a pilot run
            self.cost.record(0, 0, time.time() - t0, error=True)
            print(f"[api_describers] {self.cost.model_id} call failed: {e}")
            return coerce({}, schema)
        return coerce(extract_json(text), schema)

    def scene_batch(self, images, batch_size=1):
        from describe import SCENE_PROMPT, SCENE_SCHEMA, schema_str
        p = SCENE_PROMPT.format(schema=schema_str(SCENE_SCHEMA))
        return [self._run_one(im, p, SCENE_SCHEMA) for im in images]

    def describe_batch(self, items, batch_size=1, prompt_template=None, schema=None):
        from describe import OBJECT_PROMPT, OBJECT_SCHEMA, schema_str
        tmpl = prompt_template or OBJECT_PROMPT
        sch_dict = schema or OBJECT_SCHEMA
        sch = schema_str(sch_dict)
        return [self._run_one(crop, tmpl.format(cls=cls, schema=sch), sch_dict)
                for crop, cls in items]


class OpenAIDescriber(_ApiDescriberBase):
    def __init__(self, model_id="gpt-5.6-sol", api_key=None, max_tokens=300,
                 base_url=None, no_thinking=False):
        from openai import OpenAI
        # base_url override lets any OpenAI-compatible endpoint reuse this
        # class (DashScope/Qwen uses it via QwenApiDescriber below)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.no_thinking = no_thinking
        self.cost = CostTracker(model_id)

    def _call(self, image_bgr, prompt):
        b64 = base64.b64encode(_encode_jpeg(image_bgr)).decode()
        kwargs = {}
        if self.no_thinking:
            kwargs["reasoning_effort"] = "minimal"   # cuts billed reasoning tokens
        resp = self.client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=self.max_tokens,  # newer models reject legacy max_tokens
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


class GeminiDescriber(_ApiDescriberBase):
    def __init__(self, model_id="gemini-3.5-flash", api_key=None, max_tokens=300,
                 no_thinking=False, media_resolution=None):
        from google import genai
        self.client = genai.Client(api_key=api_key)  # falls back to GOOGLE_API_KEY env
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.no_thinking = no_thinking
        # None = the API's default vision tokenization (what every run before
        # EXP-2026-09 Part 2 used). "low"/"medium"/"high" request Google's
        # MediaResolution modes — more image tokens = finer effective
        # resolution = higher input cost. We never resize client-side either
        # way (_encode_jpeg sends the full frame).
        self.media_resolution = media_resolution
        self.cost = CostTracker(model_id)

    def _call(self, image_bgr, prompt):
        from google.genai import types
        cfg = {"max_output_tokens": self.max_tokens}
        if self.media_resolution:
            key = f"MEDIA_RESOLUTION_{self.media_resolution.upper()}"
            cfg["media_resolution"] = getattr(types.MediaResolution, key, key)
        if self.no_thinking:
            # Flash-class models accept 0; Pro-class enforces a minimum and
            # may reject 0 — if it errors, use the default (thinking on)
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=_encode_jpeg(image_bgr),
                                       mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(**cfg),
        )
        text = resp.text or ""
        u = resp.usage_metadata
        in_tok = getattr(u, "prompt_token_count", 0) or 0
        # thinking models bill reasoning separately from the visible answer —
        # count both, or the cost report undercounts what Google charges
        out_tok = ((getattr(u, "candidates_token_count", 0) or 0)
                   + (getattr(u, "thoughts_token_count", 0) or 0))
        return text, in_tok, out_tok


class ClaudeDescriber(_ApiDescriberBase):
    def __init__(self, model_id="claude-sonnet-5", api_key=None, max_tokens=300,
                 no_thinking=False):
        # no_thinking accepted for interface parity; Claude thinking is opt-in
        # and this harness never enables it, so there is nothing to turn off
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY env
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.cost = CostTracker(model_id)

    def _call(self, image_bgr, prompt):
        b64 = base64.b64encode(_encode_jpeg(image_bgr)).decode()
        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/jpeg",
                                              "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class QwenApiDescriber(OpenAIDescriber):
    """Qwen via Alibaba DashScope's OpenAI-compatible endpoint.

    Default is qwen3.7-plus — the current flagship, natively multimodal
    (Alibaba folded vision into the mainline; the older dedicated Qwen3-VL
    line still exists as qwen3-vl-plus/-flash but is a generation behind).
    Key: DASHSCOPE_API_KEY (Alibaba Cloud Model Studio). Kept API-based (not
    local weights) so the cost comparison stays token-billed and CPU-only
    like the other three vendors.
    """

    DASHSCOPE_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model_id="qwen3.7-plus", api_key=None, max_tokens=300,
                 no_thinking=False):
        import os
        super().__init__(model_id=model_id,
                         api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
                         max_tokens=max_tokens,
                         base_url=self.DASHSCOPE_INTL,
                         no_thinking=no_thinking)

    def _call(self, image_bgr, prompt):
        # DashScope's compatible mode uses the classic max_tokens param, not
        # OpenAI's newer max_completion_tokens
        b64 = base64.b64encode(_encode_jpeg(image_bgr)).decode()
        kwargs = {}
        if self.no_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}
        resp = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


ENGINES = {
    "openai": OpenAIDescriber,
    "gemini": GeminiDescriber,
    "claude": ClaudeDescriber,
    "qwenapi": QwenApiDescriber,
}


def build_engine(name: str, model_id: str | None, api_key: str | None = None,
                 max_tokens: int | None = None, no_thinking: bool = False,
                 media_resolution: str | None = None):
    cls = ENGINES[name]
    kwargs = {"api_key": api_key, "no_thinking": no_thinking}
    if model_id:
        kwargs["model_id"] = model_id
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if media_resolution:
        if name != "gemini":
            raise SystemExit("--media-resolution is a Gemini-only knob "
                             "(other vendors expose no equivalent here)")
        kwargs["media_resolution"] = media_resolution
    return cls(**kwargs)


if __name__ == "__main__":
    # --selftest: exercise the parsing/cost path with no network calls.
    class _FakeTracker(CostTracker):
        def __init__(self):
            super().__init__("selftest-model")

    class _Fake(_ApiDescriberBase):
        def __init__(self):
            self.cost = _FakeTracker()

        def _call(self, image_bgr, prompt):
            return ('{"apparent_gender": "woman"}', 120, 15)

    import numpy as np
    fake = _Fake()
    img = np.zeros((40, 40, 3), dtype="uint8")
    out = fake.describe_batch([(img, "Woman")])
    assert out[0]["apparent_gender"] == "woman", out
    report = fake.cost.report()
    assert report["calls"] == 1 and report["input_tokens"] == 120
    print("[api_describers] selftest OK:", report)
