"""Generate mobile/tests/fixtures/fallback_parity.json from the PYTHON advice-pack fallback (src/livestock_llm/advice.py).
The mobile test then requires the TypeScript offline guidance to reproduce it field for field:
    python tools/make_guidance_fixture.py
Re-run it whenever shared/advice_pack.json or the fallback logic changes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livestock_llm.advice import AdvicePack                      # noqa: E402
from livestock_vision.engine import Labels, flags_for            # noqa: E402

pack = AdvicePack(ROOT / "shared" / "advice_pack.json")
plan = Labels(ROOT / "shared" / "labels.json")
weak_path = ROOT / "models" / "yolo11x-weak-v1" / "labels.json"
weak = Labels(weak_path) if weak_path.exists() else plan


def d(cls, conf):
    return {"class": cls, "confidence": conf, "box": [1.0, 2.0, 30.0, 40.0]}


CASES = [
    ("weak: lumpy skin + healthy", weak, [d("lumpy_skin_disease", 0.9), d("healthy", 0.6)]),
    ("weak: healthy only", weak, [d("healthy", 0.8)]),
    ("weak: FMD and LSD", weak, [d("foot_and_mouth_disease", 0.8), d("lumpy_skin_disease", 0.5)]),
    ("plan: two nodules and a tick", plan, [d("skin_nodule", 0.88), d("skin_nodule", 0.81), d("tick", 0.74)]),
    ("plan: ringworm (zoonotic)", plan, [d("ringworm_patch", 0.7)]),
    ("plan: udder + nasal discharge", plan, [d("swollen_udder", 0.6), d("nasal_discharge", 0.55)]),
    ("no detections", plan, []),
    ("class the pack does not know", plan, [d("zebra", 0.9)]),
]
out = []
for name, labels, dets in CASES:
    flags = flags_for(dets, labels) if all(x["class"] in labels.by_name for x in dets) else []
    r = pack.fallback_report(dets, flags).model_dump()
    out.append({"name": name, "detections": dets, "flags": flags, "expected": {
        k: r[k] for k in ("severity", "notifiable", "zoonotic", "possible_conditions", "immediate_actions", "prevention", "follow_up_days")}
        | {"supportive_care": r["treatment"]["supportive_care"], "requires_vet": r["treatment"]["requires_vet"]}})
dest = ROOT / "mobile" / "tests" / "fixtures" / "fallback_parity.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"pack_version": pack.version, "cases": out}, indent=1), encoding="utf-8")
print(f"wrote {len(out)} cases to {dest}")
