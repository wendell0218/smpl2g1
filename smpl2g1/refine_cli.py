import argparse
import json
from pathlib import Path

from .refinement import refine_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpl-reference", type=Path)
    parser.add_argument("--upper-body-strength", type=float, default=0.75)
    parser.add_argument("--contact", action="store_true")
    args = parser.parse_args()
    report = refine_motion(args.input, args.output, args.smpl_reference, args.upper_body_strength, args.contact)
    print(json.dumps(report, indent=2))
