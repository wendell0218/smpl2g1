import argparse
import json
from pathlib import Path

from .generation import generate_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-model-dir", type=Path, required=True)
    parser.add_argument("--umr-root", type=Path, required=True)
    parser.add_argument("--umr-env", type=Path, required=True)
    parser.add_argument("--umr-slots", type=Path, required=True)
    parser.add_argument("--up-axis", choices=("y", "z"), default="y")
    args = parser.parse_args()
    report = generate_motion(
        args.input,
        args.output,
        args.body_model_dir,
        args.umr_root,
        args.umr_env,
        args.umr_slots,
        args.up_axis,
    )
    print(json.dumps(report, indent=2))
