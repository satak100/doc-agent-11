"""Governance — conservative PII detection/redaction before indexing."""
from __future__ import annotations

import re

from ..contracts import Chunk

PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?88)?01[3-9][\d -]{8,12}(?!\d)"),
    "patient_name": re.compile(r"(?im)^(\s*name\s*:\s*).+$"),
}

def detect(text: str) -> list[tuple[int,int,str]]:
    spans = []
    for kind, pattern in PATTERNS.items():
        spans.extend((match.start(), match.end(), kind) for match in pattern.finditer(text))
    return sorted(spans)


def redact(text: str) -> str:
    for kind, pattern in PATTERNS.items():
        if kind == "patient_name":
            text = pattern.sub(r"\1[REDACTED]", text)
        else:
            text = pattern.sub(f"[{kind.upper()}_REDACTED]", text)
    return text



def register(hooks) -> None:
    def _scrub(ctx: dict) -> dict:
        if "chunks" in ctx:
            ctx["chunks"] = [
                chunk.model_copy(update={"text": redact(chunk.text)})
                if isinstance(chunk, Chunk)
                else chunk
                for chunk in ctx["chunks"]
            ]
        if isinstance(ctx.get("text"), str):
            ctx["text"] = redact(ctx["text"])
        return ctx
    hooks.register(hooks.AFTER_OCR, _scrub)       # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)   # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)          # scrub logs
