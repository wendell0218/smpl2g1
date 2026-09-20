# <modify>+Command-line entry point for hybrid G1 trajectory refinement.
import argparse
import json
from pathlib import Path

from .refinement import refine_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = refine_motion(args.input, args.output)
    print(json.dumps(report, indent=2))
