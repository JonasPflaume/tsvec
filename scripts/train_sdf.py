#!/usr/bin/env python3
"""Train the neural SDF model used by TSVec."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_sdf.checkpoint import PORTABLE_CHECKPOINT_NAME, save_sdf_npz
from neural_sdf.training import fit, load_mesh, vertex_normals


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
    parser.add_argument("--mesh", default=None, help="Optional explicit OBJ mesh path.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sdf_model")
    parser.add_argument("--target-faces", type=int, default=500000)
    parser.add_argument("--unit", choices=["m", "mm"], default="m")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--checkpoint-step", type=int, default=8000)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--save-orbax", action="store_true")
    parser.add_argument("--mesh-diagnostics", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Show the learned SDF mesh after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = resolve_path(
        args.mesh,
        RELEASE_ROOT / "ergodic_dataset" / args.model / f"{args.model}.obj",
    )
    checkpoint_dir = resolve_path(args.checkpoint_dir, RELEASE_ROOT / "checkpoints" / "sdf_model")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    vertices, faces, normalization_params = load_mesh(
        str(mesh_path),
        target_faces=args.target_faces,
        unit=args.unit,
        diagnostics=args.mesh_diagnostics,
    )
    normals = vertex_normals(vertices, faces)

    print("Starting SDF training...")
    loss, sdf_state = fit(
        vertices,
        normals=normals,
        lam=args.lam,
        tau=args.tau,
        depth=args.depth,
        hidden=args.hidden,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
    )

    if args.save_orbax:
        from flax.training import checkpoints

        checkpoint_data = {
            "sdf_state": sdf_state,
            "normalization_params": normalization_params,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=str(checkpoint_dir),
            target=checkpoint_data,
            step=args.checkpoint_step,
            keep=args.keep,
            overwrite=True,
        )
    portable_checkpoint = save_sdf_npz(
        checkpoint_dir / PORTABLE_CHECKPOINT_NAME,
        sdf_state,
        normalization_params,
    )

    print(f"SDF training completed with final loss: {float(loss):.8f}")
    if args.save_orbax:
        print(f"Orbax checkpoint saved to: {checkpoint_dir}")
    print(f"Portable checkpoint saved to: {portable_checkpoint}")
    print(f"Saved normalization params: {normalization_params}")

    if args.plot:
        from neural_sdf.sdf_jax.util import plot3d

        sdf_model_fn = lambda x: sdf_state.apply_fn(
            {"params": sdf_state.params, **sdf_state.model_state}, x
        ).squeeze()
        plot3d(sdf_model_fn, ngrid=50)


if __name__ == "__main__":
    main()
