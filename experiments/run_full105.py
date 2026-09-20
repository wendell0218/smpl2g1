import argparse
import json
from pathlib import Path

from smpl2g1.refinement import refine_motion


parser = argparse.ArgumentParser()
parser.add_argument("--shard-index", type=int, required=True)
parser.add_argument("--num-shards", type=int, default=4)
args = parser.parse_args()

user_root = Path("/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user")
input_root = Path("/mnt/workspace/umr_jerk_eval/full_metrics/g1/UMR")
reference_root = user_root / "code/OmniHM_dataset/outputs/sonic_full1000_20260915/g1/OmniHM-5M"
output_root = Path("outputs/hybrid_contact_full105")
paths = sorted(input_root.glob("*.npz"))
assert len(paths) == 105

for index, path in enumerate(paths[args.shard_index::args.num_shards], 1):
    reference = reference_root / path.name
    output = output_root / path.name
    report = refine_motion(path, output, reference)
    print(json.dumps({"shard": args.shard_index, "done": index, "sample": path.stem, **report}), flush=True)
