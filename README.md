# TSVec Ergodic Planning

This folder is a standalone release for running the TSVec planner on the
included `ergodic_dataset` objects. It keeps the main workflow from the paper:
train an SDF, run TSVec, then render the planned trajectory with the
provided Open3D visualization.

## Paper and Video

- Paper: [Stein Variational Ergodic Surface Coverage with SE(3) Constraints](https://arxiv.org/abs/2603.09458)
- Video: [https://www.youtube.com/watch?v=djsHoxP5ov8](https://www.youtube.com/watch?v=djsHoxP5ov8)

## Contents

- `ergodic_dataset/`: meshes and colored point clouds for bunny, cylinder, hand,
  mustardbottle, pig, spot, and torus.
- `tsvec/`: TSVec planner, ergodic objective, metrics, and visualization code.
- `neural_sdf/`: neural SDF model, training, checkpoint, and grid utilities.
- `lbfgsb/`: point-cloud and spectral utilities used by the ergodic objective.
- `scripts/`: runnable entry points for training, planning, and visualization.

SDF checkpoints are written to `checkpoints/sdf_model/` when you run
`scripts/train_sdf.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The requirements install NVIDIA GPU JAX via `jax[cuda12]`, which bundles the
CUDA/cuDNN pip wheels. If your machine uses CUDA 13 with a sufficiently recent
driver, change that line in `requirements.txt` to `jax[cuda13]>=0.6.0`.

The requirements also install `jaxkd[cuda]`, which provides the CUDA extension
used by `--jaxkd-backend cuda`. If the extension is unavailable on a target
machine, use `--jaxkd-backend jax` as the fallback.

## Workflow

Train an SDF checkpoint first, then run the planner:

```bash
python scripts/train_sdf.py --model bunny
python scripts/run_tsvec.py --model bunny --jaxkd-backend cuda
python scripts/visualize_result.py --model bunny
```

`visualize_result.py` opens an interactive Open3D window by default. Use the
mouse to rotate/zoom the view. Add `--save-image` and press `S` in the window to
save the current view, or use `--save-only` to save one PNG and exit.

`train_sdf.py` writes a portable `sdf_model.npz`. The run and visualization
scripts load this file by default, which avoids device-sharding issues when
moving a checkpoint between CUDA and CPU machines. Add `--save-orbax` if you
also want a Flax/Orbax checkpoint for your local environment.

The planner saves:

- `outputs/<model>/xpos_sdf.npy`
- `outputs/<model>/quat_sdf.npy`
- `outputs/<model>/xpos_sdf_all.npy`
- `outputs/<model>/quat_sdf_all.npy`
- `outputs/<model>/particle_logll.npy`
- `outputs/<model>/run_config.json`

Add `--save-history` if you also want per-iteration histories
(`xpos_his_sdf.npy` and `quat_his_sdf.npy`). They are large and are skipped by
default because visualization only needs the final trajectory arrays.

Open3D rendering opens a window and captures a PNG in the result directory. On a
headless machine, run visualization through a display server or skip rendering
and inspect the saved NumPy arrays.

## Quick Installation Check

These settings are only for checking that the environment works:

```bash
python scripts/run_tsvec.py --model torus --iterations 2 --warmup-iterations 1 \
  --particles 2 --horizon 40 --jaxkd-backend cuda
```

For publication-quality runs, use the default planning parameters or the
parameters reported with the released experiment.

## Citation

This repository accompanies the paper above, accepted to IEEE ICRA 2026.

```bibtex
@inproceedings{li2026stein,
  title     = {Stein Variational Ergodic Surface Coverage with {SE}(3) Constraints},
  author    = {Li, Jiayun and Jin, Yufeng and Teng, Sangli and Gong, Dejian and Chalvatzaki, Georgia},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  note      = {Accepted, to appear. arXiv:2603.09458},
  doi       = {10.48550/arXiv.2603.09458},
}
```
