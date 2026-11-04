"""The vet advice pack: the offline guidance and the API's fallback when the language model cannot answer."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import Report, Treatment

URGENCY_RANK = {"none": 0, "soon": 1, "urgent": 2, "emergency": 3}
SEVERITY = {"none": "low", "soon": "moderate", "urgent": "high", "emergency": "urgent"}
FOLLOW_UP = {"none": None, "soon": 7, "urgent": 2, "emergency": 1}


def _dedupe(items, cap):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out[:cap]


class AdvicePack:
    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.version = raw.get("data_version")
        self.reviewed = bool(raw.get("reviewed_by_veterinarian"))
        self.disclaimer = raw.get("disclaimer", "")
        self.entries = raw["diseases"]
        self.by_key = {k.lower(): e for e in self.entries for k in [e["label"], *e.get("aliases", [])]}

    def lookup(self, cls: str) -> dict | None:
        return self.by_key.get(cls.lower())

    def for_detections(self, dets: list[dict]) -> list[dict]:
        """Advice entries for the classes present, most confident first, without duplicates."""
        out, seen = [], set()
        for d in sorted(dets, key=lambda d: -d.get("confidence", 0)):
            e = self.lookup(d["class"])
            if e and e["label"] not in seen:
                seen.add(e["label"])
                out.append(e)
        return out

    def fallback_report(self, dets: list[dict], flags: list[str]) -> Report:
        """A complete report from approved text only: what users get when the LLM is down, slow or invalid."""
        entries = self.for_detections(dets)
        findings = [e for e in entries if e["label"] != "healthy"] or entries[:1]
        counts: dict[str, int] = {}
        for d in dets:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        top = max((URGENCY_RANK[e["urgency"]] for e in findings), default=0)
        urgency = {v: k for k, v in URGENCY_RANK.items()}[top]
        summary = ("Findings: " + ", ".join(f"{n} x {c.replace('_', ' ')}" for c, n in counts.items()) + ". "
                   if counts else "No finding was detected in this photo. ")
        summary += "This is standard guidance from the veterinary advice pack; the AI-written report is not available right now."
        return Report(
            summary=summary[:800],
            possible_conditions=[e["title"] for e in findings] if findings and findings[0]["label"] != "healthy"
            else ["No condition recognised in this photo"],
            severity=SEVERITY[urgency],
            notifiable=any(e.get("notifiable") for e in findings) or "possible_notifiable_disease" in flags,
            zoonotic=any(e.get("zoonotic") for e in findings) or "possible_zoonotic_risk" in flags,
            immediate_actions=_dedupe([a for e in findings or entries for a in e["do_now"]], 10),
            treatment=Treatment(supportive_care=_dedupe([e["see_vet"] for e in findings or entries], 6), medicines=[],
                                requires_vet=True, withdrawal_periods=[]),
            prevention=_dedupe([p for e in findings or entries for p in e["prevention"]], 8),
            follow_up_days=FOLLOW_UP[urgency],
        )
