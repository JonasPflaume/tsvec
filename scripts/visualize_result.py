#!/usr/bin/env python3
"""Render a saved TSVec trajectory using the release visualization code."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return RELEASE_ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="torus", help="Dataset name under ergodic_dataset/.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sdf_model")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--number", default="result")
    parser.add_argument("--sdf-resolution", type=int, default=64)
    parser.add_argument("--surface-only", action="store_true")
    parser.add_argument("--no-sdf", action="store_true")
    parser.add_argument("--save-image", action="store_true", help="Enable S-key screenshot saving in interactive mode.")
    parser.add_argument("--save-only", action="store_true", help="Save one PNG and exit without opening an interactive session.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_path(args.checkpoint_dir, RELEASE_ROOT / "checkpoints" / "sdf_model")
    result_dir = resolve_path(args.result_dir, RELEASE_ROOT / "outputs" / args.model)
    output_dir = resolve_path(args.output_dir, result_dir)
    os.environ["TSVEC_SDF_CHECKPOINT"] = str(checkpoint_dir)

    import jax.numpy as jnp
    from jaxlie import SO3

    from helper.point_cloud_utils import process_point_cloud_match_sdf
    from neural_sdf.checkpoint import load_sdf_model
    from tsvec.param import Param
    from tsvec.planner import TSVecState, config as tsvec_config
    from tsvec.visualize import visualize_trajectory

    loaded_model_fn, normalization_params = load_sdf_model()

    dataset_path = RELEASE_ROOT / "ergodic_dataset" / args.model / f"{args.model}_colored.ply"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Colored point cloud not found: {dataset_path}")

    ext_param = Param()
    pcloud = process_point_cloud_match_sdf(str(dataset_path), ext_param, normalization_params)

    state = None
    if not args.surface_only:
        pos_path = result_dir / "xpos_sdf.npy"
        quat_path = result_dir / "quat_sdf.npy"
        if not pos_path.exists() or not quat_path.exists():
            raise FileNotFoundError(
                f"Missing trajectory arrays in {result_dir}. Run scripts/run_tsvec.py first."
            )
        positions = jnp.array(np.load(pos_path))
        rotations = jnp.array(np.load(quat_path))
        run_config = dict(tsvec_config)
        run_config["particle_num"] = int(positions.shape[0])
        run_config["horizon"] = int(positions.shape[1])
        init_points = jnp.stack([positions[0, 0], positions[0, -1]], axis=0)
        state = TSVecState.create(init_points, run_config).replace(
            pos_particles=positions,
            rot_particles=SO3(rotations),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (str(args.number) + ".png")
    old_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        visualize_trajectory(
            state,
            pcloud,
            sdf_fn=None if args.no_sdf else loaded_model_fn,
            sdf_bounds=None if args.no_sdf else [[0, 1], [0, 1], [0, 1]],
            sdf_resolution=args.sdf_resolution,
            number=args.number,
            interactive=not args.save_only,
            save_image=args.save_image or args.save_only,
            output_path=output_path,
        )
    finally:
        os.chdir(old_cwd)

    if args.save_only:
        print(f"Saved render to: {output_path}")


if __name__ == "__main__":
    main()
