#!/usr/bin/env python3
"""
Convert one of our self-generated report HTML files to GitHub-flavored
Markdown (headings, paragraphs, lists, tables). Built for the deploycmp
reports (women_report_final.py etc.) — simple, style-free HTML in, one
.md out. Inline SVG blocks are dropped with a placeholder line.

    python3 report_to_md.py WOMEN_REPORT_FINAL.html [out.md]
    python3 report_to_md.py --selftest
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class MD(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.buf = []
        self.mode = []            # element stack
        self.row = []
        self.table = []
        self.svg_depth = 0

    def _flush_text(self):
        txt = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        return txt

    def handle_starttag(self, tag, attrs):
        if tag == "svg" or self.svg_depth:
            self.svg_depth += 1
            return
        if tag in ("h1", "h2", "h3", "p", "li", "table", "tr", "ul", "title"):
            self.mode.append(tag)
            if tag == "table":
                self.table = []
            if tag == "tr":
                self.row = []
            self.buf = []
        elif tag in ("td", "th"):
            self.buf = []
        elif tag == "b" or tag == "i":
            self.buf.append("**" if tag == "b" else "*")
        elif tag == "br":
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if self.svg_depth:
            if tag == "svg":
                self.svg_depth -= 1
                if self.svg_depth == 0:
                    self.out.append("*(chart omitted in the Markdown "
                                    "version — see the HTML report)*\n")
            return
        if tag in ("b", "i"):
            self.buf.append("**" if tag == "b" else "*")
        elif tag in ("td", "th"):
            self.row.append(self._flush_text().replace("|", "\\|"))
        elif tag == "tr":
            self.table.append(self.row)
            self.row = []
            if self.mode and self.mode[-1] == "tr":
                self.mode.pop()
        elif tag == "table":
            if self.table:
                head, *body = self.table
                w = len(head)
                lines = ["| " + " | ".join(head) + " |",
                         "|" + "---|" * w]
                for r in body:
                    r = (r + [""] * w)[:w]
                    lines.append("| " + " | ".join(r) + " |")
                self.out.append("\n".join(lines) + "\n")
            while self.mode and self.mode[-1] in ("table", "tr"):
                self.mode.pop()
        elif tag in ("h1", "h2", "h3", "p", "li", "title"):
            txt = self._flush_text()
            if self.mode and self.mode[-1] == tag:
                self.mode.pop()
            if not txt:
                return
            if tag == "h1":
                self.out.append(f"# {txt}\n")
            elif tag == "h2":
                self.out.append(f"## {txt}\n")
            elif tag == "h3":
                self.out.append(f"### {txt}\n")
            elif tag == "li":
                self.out.append(f"- {txt}")
            elif tag == "p":
                self.out.append(f"{txt}\n")
        elif tag == "ul":
            self.out.append("")
            if self.mode and self.mode[-1] == "ul":
                self.mode.pop()

    def handle_data(self, data):
        if self.svg_depth:
            return
        # ignore text with no open block (style/script are skipped by state)
        if self.mode and self.mode[-1] in ("h1", "h2", "h3", "p", "li",
                                           "title") or self.row is not None:
            self.buf.append(data)

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.buf.append(" ")


def convert(html_text: str) -> str:
    # strip style/script blocks entirely first
    html_text = re.sub(r"<style.*?</style>", "", html_text, flags=re.S)
    html_text = re.sub(r"<script.*?</script>", "", html_text, flags=re.S)
    m = MD()
    m.feed(html_text)
    return "\n".join(m.out).strip() + "\n"


def selftest():
    html = """<!doctype html><style>x{}</style><title>T</title>
<h1>Head</h1><p>Para with <b>bold</b>.</p>
<ul><li>item one</li><li>item two</li></ul>
<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2|3</td></tr></table>
<svg><rect/></svg><h2>Sec</h2>"""
    md = convert(html)
    assert "# Head" in md and "**bold**" in md, md
    assert "- item one" in md, md
    assert "| a | b |" in md and "| 1 | 2\\|3 |" in md, md
    assert "chart omitted" in md and "<svg" not in md, md
    assert "## Sec" in md, md
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".md")
    dst.write_text(convert(src.read_text()))
    print(f"[md] wrote {dst}")
