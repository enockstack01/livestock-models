"""The report contract and the veterinary safety rules, enforced IN CODE (a prompt is a request; this is a guarantee).

Rules from the Logistack plan:
  * doses are only given when the animal's weight is provided; prescription-style medicines are marked "use under veterinary supervision"
  * every medicine recommendation carries milk and meat withdrawal periods
  * suspected notifiable disease -> isolate + contact the sector veterinary officer / RAB; the record is flagged
  * suspected zoonosis -> handling and hygiene warnings
  * findings are signs, not diagnoses: the report always recommends veterinary confirmation
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DISCLAIMER = ("This is decision support based on visible signs in a photo, not a veterinary diagnosis. "
              "Confirm serious or notifiable conditions with a veterinarian or a laboratory test.")
DOSE_REMOVED = "[dose: needs the animal's weight and a veterinarian]"
SUPERVISION = "use under veterinary supervision"
ISOLATE = "Isolate the animal from the rest of the herd and do not move animals on or off the farm."
CONTACT_VET = "Contact the sector veterinary officer / RAB immediately."
HYGIENE = "Wear gloves and wash hands and clothes after handling the animal: this condition can spread to people."
WITHDRAWAL = ("Milk and meat withdrawal periods apply to any treated animal: follow the product label and ask the "
              "veterinarian before using milk or meat.")
SEVERITY_ORDER = ["low", "moderate", "high", "urgent"]

# a number followed by a dose unit, optionally "/kg", "per head" ... (10 mg/kg, 2 ml, 500 mg per animal)
_DOSE = re.compile(r"(?i)\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|kg|ml|l|cc|iu|units?)\b(?:\s*(?:/|per)\s*(?:kg|animal|head|day|dose|ml|litre|liter))?")


def _str(v: str) -> str:
    return " ".join(str(v).split())[:500]


class Treatment(BaseModel):
    supportive_care: list[str] = Field(default_factory=list, max_length=12)
    medicines: list[str] = Field(default_factory=list, max_length=10)
    requires_vet: bool = True
    withdrawal_periods: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("supportive_care", "medicines", "withdrawal_periods", mode="before")
    @classmethod
    def clean(cls, v):
        return [_str(x) for x in v if str(x).strip()] if isinstance(v, list) else v


class Report(BaseModel):
    summary: str = Field(max_length=800)
    possible_conditions: list[str] = Field(default_factory=list, max_length=8)
    severity: Literal["low", "moderate", "high", "urgent"] = "moderate"
    notifiable: bool = False
    zoonotic: bool = False
    immediate_actions: list[str] = Field(default_factory=list, max_length=12)
    treatment: Treatment = Field(default_factory=Treatment)
    prevention: list[str] = Field(default_factory=list, max_length=10)
    follow_up_days: int | None = Field(default=None, ge=0, le=60)
    disclaimer: str | None = None

    @field_validator("possible_conditions", "immediate_actions", "prevention", mode="before")
    @classmethod
    def clean(cls, v):
        return [_str(x) for x in v if str(x).strip()] if isinstance(v, list) else v


class AnimalContext(BaseModel):
    weight_kg: float | None = Field(default=None, gt=0, le=2000)
    age_months: int | None = Field(default=None, ge=0, le=600)
    lactating: bool | None = None
    pregnant: bool | None = None


def report_json_schema() -> dict:
    return Report.model_json_schema()


def sanitize_report(report: Report, flags: list[str], animal: AnimalContext | None) -> tuple[Report, list[str]]:
    """Apply the safety rules to a model-written report. Returns the safe report and a list of what was changed."""
    adj: list[str] = []
    r = report.model_copy(deep=True)
    has_weight = bool(animal and animal.weight_kg)

    if "possible_notifiable_disease" in flags:
        if not r.notifiable:
            r.notifiable = True
            adj.append("notifiable flag forced on (the detector flagged a possible notifiable disease)")
        if SEVERITY_ORDER.index(r.severity) < SEVERITY_ORDER.index("high"):
            r.severity = "high"
            adj.append("severity raised to 'high' for a possible notifiable disease")
    if r.notifiable:
        if not any(re.search(r"(?i)isolat", a) for a in r.immediate_actions):
            r.immediate_actions.insert(0, ISOLATE)
            adj.append("added isolation advice")
        if not any(re.search(r"(?i)veterinar|\bRAB\b", a) for a in r.immediate_actions):
            r.immediate_actions.insert(1 if len(r.immediate_actions) > 0 else 0, CONTACT_VET)
            adj.append("added 'contact the sector veterinary officer / RAB'")
    if "possible_zoonotic_risk" in flags and not r.zoonotic:
        r.zoonotic = True
        adj.append("zoonotic flag forced on")
    if r.zoonotic and not any(re.search(r"(?i)glove|wash|hygien", a) for a in r.immediate_actions):
        r.immediate_actions.append(HYGIENE)
        adj.append("added zoonotic hygiene warning")

    meds = []
    for m in r.treatment.medicines:
        if not has_weight and _DOSE.search(m):
            m = _DOSE.sub(DOSE_REMOVED, m)
            adj.append("removed a dose: the animal's weight was not provided")
        if SUPERVISION not in m.lower():                      # the exact phrase: the dose placeholder also says "veterinarian"
            m = f"{m} ({SUPERVISION})"
        meds.append(m)
    r.treatment.medicines = meds
    if meds:
        if not r.treatment.withdrawal_periods:
            r.treatment.withdrawal_periods = [WITHDRAWAL]
            adj.append("added milk/meat withdrawal advice")
        if has_weight:
            adj.append("doses (if any) must follow the product label and veterinary instruction")
    if r.notifiable or r.severity in ("high", "urgent") or meds:
        r.treatment.requires_vet = True
    r.disclaimer = DISCLAIMER
    return r, adj
