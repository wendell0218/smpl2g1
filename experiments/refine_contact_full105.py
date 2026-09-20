# <modify>+Run the fixed contact-refinement configuration on one deterministic full105 shard.
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
BASE = REPO.parents[1]
OMG = BASE / "code/OMG"
sys.path.insert(0, str(OMG / "src"))
from omg.robots.g1.kinematics import G1Kinematics
contact_spec = importlib.util.spec_from_file_location("contact_refinement", REPO / "smpl2g1/contact_refinement.py")
contact_module = importlib.util.module_from_spec(contact_spec)
contact_spec.loader.exec_module(contact_module)
refine_contact_motion = contact_module.refine_contact_motion


parser = argparse.ArgumentParser()
parser.add_argument("--shard-index", type=int, required=True)
parser.add_argument("--num-shards", type=int, default=4)
parser.add_argument("--pilot", action="store_true")
args = parser.parse_args()
input_root = REPO / "outputs/hybrid_upper_full105"
output_root = REPO / ("outputs/hybrid_contact_pilot" if args.pilot else "outputs/hybrid_contact_full105")
paths = sorted(input_root.glob("*.npz"))
if args.pilot:
    pilot_ids = {"omnihm_0102", "omnihm_0097", "omnihm_0089", "omnihm_0065", "omnihm_0043", "omnihm_0121",
                 "omnihm_0039", "omnihm_0051", "omnihm_0026", "omnihm_0068", "omnihm_0053", "omnihm_0079"}
    paths = [path for path in paths if path.stem in pilot_ids]
paths = paths[args.shard_index::args.num_shards]
config = json.loads((REPO / "smpl2g1/configs/g1_limits.json").read_text())
limits = np.column_stack([config["lower"], config["upper"]]).astype(float)
velocity = np.asarray(config["velocity"], dtype=float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kinematics = G1Kinematics(OMG / "assets/robots/g1/g1_kinematics.json").to(device).eval()

for index, path in enumerate(paths, 1):
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos_36"], np.float64)
        valid = np.asarray(data["valid"], bool)
        metadata = {key: data[key] for key in data.files if key != "qpos_36"}
    output, report = refine_contact_motion(qpos, valid, kinematics, limits, velocity)
    destination = output_root / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, qpos_36=output.astype(np.float32), **metadata)
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"shard": args.shard_index, "done": index, "total": len(paths),
                      "sample": path.stem, **report}), flush=True)
