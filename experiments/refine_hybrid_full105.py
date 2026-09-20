# <modify>+Generate the complete paired cohort with the validated hybrid refinement.
import json
from pathlib import Path

from smpl2g1.refinement import refine_motion


INPUT_ROOT = Path("/mnt/workspace/umr_jerk_eval/full_metrics/g1/UMR")
OUTPUT_ROOT = Path("outputs/hybrid_rootpose_full105")
paths = sorted(INPUT_ROOT.glob("*.npz"))
assert len(paths) == 105
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
reports = []
for index, path in enumerate(paths):
    report = refine_motion(path, OUTPUT_ROOT / path.name)
    report["sample_id"] = path.stem
    reports.append(report)
    print(f"[{index + 1}/{len(paths)}] {path.stem}", flush=True)
(OUTPUT_ROOT / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
