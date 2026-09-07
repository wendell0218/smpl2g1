import argparse
import json
from pathlib import Path

from .pipeline import retarget_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-model-dir", type=Path, required=True)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--up-axis", choices=("y", "z"), default="z")
    parser.add_argument("--ground-height", type=float, default=0.005)
    args = parser.parse_args()
    report = retarget_motion(
        input_path=args.input,
        output_path=args.output,
        body_model_dir=args.body_model_dir,
        target_fps=args.target_fps,
        up_axis=args.up_axis,
        valid_path=args.valid,
        ground_height=args.ground_height,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))
