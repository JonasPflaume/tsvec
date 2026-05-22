#!/usr/bin/env python3
"""Run TSVec trajectory optimization on an ergodic_dataset object."""

from __future__ import annotations

import argparse
import json
import os
import time
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--init-ker-l", type=float, default=0.05)
    parser.add_argument("--init-noise", type=float, default=0.005)
    parser.add_argument("--init-lr", type=float, default=0.1)
    parser.add_argument("--smoothness-weight", type=float, default=5.0)
    parser.add_argument("--ergodic-weight", type=float, default=0.1)
    parser.add_argument("--convergence-tol", type=float, default=1e-8)
    parser.add_argument(
        "--particle-index",
        default="best",
        help="Particle to save: 'best' or an integer index. Use 0 for the old particle-0 behavior.",
    )
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--jaxkd-backend", choices=["cuda", "jax"], default="cuda")
    parser.add_argument("--render-final", action="store_true")
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--sdf-resolution", type=int, default=64)
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save per-iteration particle histories. This is useful for debugging but expensive.",
    )
    parser.add_argument("--log-every", type=int, default=1, help="Print progress every N iterations. Use 0 to disable.")
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace, checkpoint_dir: Path) -> None:
    if args.jax_platform != "auto":
        os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    os.environ["TSVEC_SDF_CHECKPOINT"] = str(checkpoint_dir)
    os.environ["TSVEC_JAXKD_CUDA"] = "1" if args.jaxkd_backend == "cuda" else "0"


def render_state(state, pcloud, sdf_fn, output_dir: Path, number, sdf_resolution: int) -> None:
    from tsvec.visualize import visualize_trajectory

    output_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        visualize_trajectory(
            state,
            pcloud,
            sdf_fn=sdf_fn,
            sdf_bounds=[[0, 1], [0, 1], [0, 1]],
            sdf_resolution=sdf_resolution,
            number=number,
        )
    finally:
        os.chdir(old_cwd)


def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_path(args.checkpoint_dir, RELEASE_ROOT / "checkpoints" / "sdf_model")
    output_dir = resolve_path(args.output_dir, RELEASE_ROOT / "outputs" / args.model)
    configure_runtime(args, checkpoint_dir)

    import jax
    jax.config.update("jax_enable_x64", False)
    if args.jax_platform != "auto":
        jax.config.update("jax_platform_name", args.jax_platform)
    import jax.numpy as jnp
    from jaxlie import SO3

    from lbfgsb.point_cloud_utils import process_point_cloud_match_sdf
    from neural_sdf.checkpoint import load_sdf_model
    from tsvec.ergodic import ErgodicProblem
    from tsvec.param import Param
    from tsvec.planner import TSVecPlanner, TSVecProblem, TSVecState, config as tsvec_config

    loaded_model_fn, normalization_params = load_sdf_model()

    dataset_path = RELEASE_ROOT / "ergodic_dataset" / args.model / f"{args.model}_colored.ply"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Colored point cloud not found: {dataset_path}")

    ext_param = Param()
    pcloud = process_point_cloud_match_sdf(str(dataset_path), ext_param, normalization_params)
    print(f"pcloud.dt: {pcloud.dt:.3e}, h: {getattr(pcloud, 'h', None)}")

    run_config = dict(tsvec_config)
    run_config["particle_num"] = args.particles
    run_config["init_ker_l"] = args.init_ker_l
    run_config["init_noise"] = args.init_noise
    run_config["init_lr"] = args.init_lr
    run_config["smoothness_weight"] = args.smoothness_weight
    run_config["horizon"] = args.horizon
    run_config["ergodic_weight"] = args.ergodic_weight

    green_mask = pcloud.colors[:, 1] > 0.1
    selected_pcd = pcloud.vertices[green_mask]
    if selected_pcd.shape[0] < 2:
        raise ValueError("Need at least two green goal points in the colored point cloud.")

    distances = jnp.linalg.norm(selected_pcd[:, None, :] - selected_pcd[None, :, :], axis=2)
    max_dist_idx = jnp.unravel_index(jnp.argmax(distances), distances.shape)
    projected_points = selected_pcd[jnp.array(max_dist_idx)]

    ergodic_problem = ErgodicProblem.create(pcloud, ext_param, run_config["horizon"])
    init_state = TSVecState.create(projected_points, run_config)
    problem = TSVecProblem.create(run_config, loaded_model_fn, ergodic_problem)
    planner = TSVecPlanner()
    jitted_inner_loop = jax.jit(planner.inner_loop)

    if args.warmup_iterations > 0:
        key = jax.random.PRNGKey(1)
        sample_indices = jax.random.choice(key, pcloud.vertices.shape[0], (2,), replace=False)
        warmup_points = pcloud.vertices[sample_indices]
        warmup_state = TSVecState.create(warmup_points, run_config)
        start = time.time()
        for i in range(args.warmup_iterations):
            print("Warmup iteration:", i)
            warmup_state = jitted_inner_loop(warmup_state, problem)
            print(warmup_state.curr_logll.min())
        print("Warmup took:", time.time() - start)

    pos_his = []
    rot_his = []
    best_logll = float("inf")
    start = time.time()
    completed_iterations = 0

    for i in range(args.iterations):
        should_log = args.log_every > 0 and i % args.log_every == 0
        if should_log:
            print("Iteration:", i)
        init_state = jitted_inner_loop(init_state, problem)
        logll_stats = jax.device_get(jnp.stack([jnp.min(init_state.curr_logll), jnp.mean(init_state.curr_logll)]))
        min_logll = float(logll_stats[0])
        mean_logll = float(logll_stats[1])
        if should_log:
            print(min_logll)

        if args.save_history:
            pos_his.append(np.asarray(init_state.pos_particles.reshape(1, -1)))
            rot_his.append(np.asarray(init_state.rot_particles.wxyz.reshape(1, -1)))

        if args.render_every > 0 and i % args.render_every == 0:
            render_state(init_state, pcloud, loaded_model_fn, output_dir, i, args.sdf_resolution)

        completed_iterations = i + 1
        if abs(mean_logll - best_logll) < args.convergence_tol:
            print("Converged.")
            break
        best_logll = mean_logll

    print("Planning took:", time.time() - start)

    if args.particle_index == "best":
        particle_index = int(jnp.argmin(init_state.curr_logll))
    else:
        particle_index = int(args.particle_index)
    if particle_index < 0 or particle_index >= init_state.particle_num:
        raise IndexError(f"particle-index must be in [0, {init_state.particle_num - 1}]")

    final_sdf_values = jax.vmap(jax.vmap(loaded_model_fn))(init_state.pos_particles)
    selected_sdf_abs = jnp.abs(final_sdf_values[particle_index])
    print(
        "Selected particle:",
        particle_index,
        "objective:",
        float(init_state.curr_logll[particle_index]),
        "mean |SDF|:",
        float(jnp.mean(selected_sdf_abs)),
        "max |SDF|:",
        float(jnp.max(selected_sdf_abs)),
    )

    best_state = init_state.replace(
        pos_particles=init_state.pos_particles[particle_index:particle_index + 1],
        rot_particles=SO3(init_state.rot_particles.wxyz[particle_index:particle_index + 1]),
        kappa=init_state.kappa[particle_index:particle_index + 1],
        v0=init_state.v0[particle_index:particle_index + 1],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "xpos_sdf.npy", np.asarray(best_state.pos_particles))
    np.save(output_dir / "quat_sdf.npy", np.asarray(best_state.rot_particles.wxyz))
    np.save(output_dir / "xpos_sdf_all.npy", np.asarray(init_state.pos_particles))
    np.save(output_dir / "quat_sdf_all.npy", np.asarray(init_state.rot_particles.wxyz))
    np.save(output_dir / "particle_logll.npy", np.asarray(init_state.curr_logll))
    np.save(output_dir / "particle_sdf_abs.npy", np.asarray(jnp.abs(final_sdf_values)))
    if args.save_history and pos_his:
        np.save(output_dir / "xpos_his_sdf.npy", np.concatenate(pos_his, axis=0))
        np.save(output_dir / "quat_his_sdf.npy", np.concatenate(rot_his, axis=0))
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "checkpoint_dir": str(checkpoint_dir),
                "iterations": completed_iterations,
                "config": run_config,
                "jaxkd_backend": args.jaxkd_backend,
                "selected_particle_index": particle_index,
                "save_history": args.save_history,
                "log_every": args.log_every,
            },
            f,
            indent=2,
        )

    print(f"Saved trajectory arrays to: {output_dir}")

    if args.render_final:
        render_state(best_state, pcloud, loaded_model_fn, output_dir, "final", args.sdf_resolution)


if __name__ == "__main__":
    main()
