# <modify>+Command-line entry point for hybrid G1 trajectory refinement.
import argparse
import json
from pathlib import Path

from .refinement import refine_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # <modify>+Expose the validated optional upper-body tracking prior.
    parser.add_argument("--smpl-reference", type=Path)
    parser.add_argument("--upper-body-strength", type=float, default=0.75)
    args = parser.parse_args()
    # report = refine_motion(args.input, args.output)
    # <modify>+Forward the optional reference without changing the original two-path usage.
    report = refine_motion(args.input, args.output, args.smpl_reference, args.upper_body_strength)
    print(json.dumps(report, indent=2))
