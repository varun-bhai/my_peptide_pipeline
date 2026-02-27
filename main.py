#!/usr/bin/env python3
"""
main.py

Master orchestrator for the modified-peptide structure pipeline.

Execution order:
1) Parse sequence + modifications mapping
2) Predict backbone (af2_env)
3) Predict modified side-chains (etflow_env)
4) Stitch side-chains into backbone
5) Minimize final stitched structure
"""

import argparse
import os
import subprocess
import sys
from typing import List


def print_banner(message: str) -> None:
    """Print a highly visible log banner for pipeline progress."""
    line = "=" * 80
    print(f"\n{line}\n[INFO] {message}\n{line}", flush=True)


def run_step(step_name: str, cmd: List[str], cwd: str) -> None:
    """
    Run one pipeline step.

    Uses subprocess.run(..., check=True) so failures immediately stop the pipeline.
    """
    print_banner(step_name)
    print(f"[INFO] Running command: {' '.join(cmd)}", flush=True)

    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    """CLI entry point for the full 5-step peptide pipeline."""
    parser = argparse.ArgumentParser(
        description="Run full modified-peptide structure prediction pipeline."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Input peptide sequence (e.g., APG(5PG)APG)",
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to modifications JSON dictionary",
    )

    args = parser.parse_args()

    # Resolve key paths.
    project_root = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.abspath(args.json)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Modifications JSON file not found: {json_path}")

    # Absolute script paths (ensures pipeline works even when launched elsewhere).
    step1_script = os.path.join(project_root, "01_parse_input.py")
    step2_script = os.path.join(project_root, "02_run_backbone.py")
    step3_script = os.path.join(project_root, "03_run_sidechains.py")
    step4_script = os.path.join(project_root, "04_stitch.py")
    step5_script = os.path.join(project_root, "05_minimize.py")

    for script_path in [step1_script, step2_script, step3_script, step4_script, step5_script]:
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Required pipeline script not found: {script_path}")

    try:
        # ---------------------------------------------------------------------
        # Step 1: Parse input sequence and generate CAA FASTA + modifications map
        # ---------------------------------------------------------------------
        run_step(
            "Starting Step 1/5: Parsing input sequence and mapping modifications",
            [
                "python",
                step1_script,
                "--sequence",
                args.sequence,
                "--json",
                json_path,
            ],
            cwd=project_root,
        )

        # ---------------------------------------------------------------------
        # Step 2: Backbone prediction in AlphaFold2 environment
        # ---------------------------------------------------------------------
        run_step(
            "Starting Step 2/5: Running backbone prediction in conda env 'af2_env'",
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "af2_env",
                "python",
                step2_script,
                "--fasta",
                "input/parsed_sequence.fasta",
            ],
            cwd=project_root,
        )

        # ---------------------------------------------------------------------
        # Step 3: Side-chain prediction in ETFlow environment
        # ---------------------------------------------------------------------
        run_step(
            "Starting Step 3/5: Running side-chain generation in conda env 'etflow_env'",
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "etflow_env",
                "python",
                step3_script,
                "--mods",
                "input/modifications.txt",
            ],
            cwd=project_root,
        )

        # ---------------------------------------------------------------------
        # Step 4: Stitch side-chains into backbone
        # ---------------------------------------------------------------------
        run_step(
            "Starting Step 4/5: Stitching modified residues into backbone",
            [
                "python",
                step4_script,
            ],
            cwd=project_root,
        )

        # ---------------------------------------------------------------------
        # Step 5: Minimize final stitched structure
        # ---------------------------------------------------------------------
        run_step(
            "Starting Step 5/5: Minimizing stitched structure with OpenBabel",
            [
                "python",
                step5_script,
            ],
            cwd=project_root,
        )

        print_banner("Pipeline completed successfully")
        print("[INFO] Final minimized structure: output/final_minimized.pdb", flush=True)

    except subprocess.CalledProcessError as exc:
        # CalledProcessError means one subprocess exited non-zero.
        print_banner("Pipeline failed")
        print(
            f"[ERROR] A step failed with exit code {exc.returncode}. "
            "Pipeline has stopped.",
            file=sys.stderr,
            flush=True,
        )
        raise

    except FileNotFoundError as exc:
        # Common cases: missing 'conda', missing OpenBabel binary, missing script/input.
        print_banner("Pipeline failed")
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
