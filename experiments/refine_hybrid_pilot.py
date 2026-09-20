# <modify>+Run the fixed, motion-stratified hybrid tuning cohort in one persistent process.
import json
from pathlib import Path

from smpl2g1.refinement import refine_motion


UMR_ROOT = Path("/mnt/workspace/umr_jerk_eval/full_metrics/g1/UMR")
OUTPUT_ROOT = Path("outputs/hybrid_pilot_feasible")
SAMPLE_IDS = [
    "omnihm_0102", "omnihm_0097", "omnihm_0089", "omnihm_0065",
    "omnihm_0043", "omnihm_0121", "omnihm_0039", "omnihm_0051",
    "omnihm_0026", "omnihm_0068", "omnihm_0053", "omnihm_0079",
]


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
reports = []
for index, sample_id in enumerate(SAMPLE_IDS):
    report = refine_motion(UMR_ROOT / f"{sample_id}.npz", OUTPUT_ROOT / f"{sample_id}.npz")
    report["sample_id"] = sample_id
    reports.append(report)
    print(f"[{index + 1}/{len(SAMPLE_IDS)}] {sample_id} rms={report['qpos_rms_change']:.6f}", flush=True)
(OUTPUT_ROOT / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
