# <modify>+Generate the complete paired cohort with the fixed upper-body tracking prior.
import json
from pathlib import Path

from smpl2g1.refinement import refine_motion


BASE = Path("/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user")
INPUT_ROOT = Path("/mnt/workspace/umr_jerk_eval/full_metrics/g1/UMR")
REFERENCE_ROOT = BASE / "code/OmniHM_dataset/outputs/sonic_full1000_20260915/g1/OmniHM-5M"
OUTPUT_ROOT = Path("outputs/hybrid_upper_full105")
paths = sorted(INPUT_ROOT.glob("*.npz"))
assert len(paths) == 105
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
reports = []
for index, path in enumerate(paths):
    reference = REFERENCE_ROOT / path.name
    report = refine_motion(path, OUTPUT_ROOT / path.name, reference, upper_body_strength=0.75)
    report["sample_id"] = path.stem
    reports.append(report)
    print(f"[{index + 1}/{len(paths)}] {path.stem}", flush=True)
(OUTPUT_ROOT / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
