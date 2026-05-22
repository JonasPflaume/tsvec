from pathlib import Path
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.training import checkpoints
from flax.traverse_util import flatten_dict, unflatten_dict
from neural_sdf.models import IGRModel, softplus

jax.config.update("jax_default_matmul_precision", "float32")

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = RELEASE_ROOT / "checkpoints" / "sdf_model"
PORTABLE_CHECKPOINT_NAME = "sdf_model.npz"


def _checkpoint_dir(ckpt_dir=None) -> Path:
    value = ckpt_dir or os.environ.get("TSVEC_SDF_CHECKPOINT") or DEFAULT_CHECKPOINT_DIR
    return Path(value).expanduser().resolve()


def _portable_checkpoint_path(ckpt_dir: Path) -> Path:
    if ckpt_dir.suffix == ".npz":
        return ckpt_dir
    return ckpt_dir / PORTABLE_CHECKPOINT_NAME


def save_sdf_npz(path, sdf_state, normalization_params) -> Path:
    """Save the trained SDF parameters in a portable NumPy format."""
    resolved_path = Path(path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    flat_params = flatten_dict(unfreeze(sdf_state.params), sep="/")
    arrays = {f"params/{key}": np.asarray(value) for key, value in flat_params.items()}
    arrays["normalization/min_coords"] = np.asarray(normalization_params["min_coords"])
    arrays["normalization/max_coord"] = np.asarray(normalization_params["max_coord"])
    np.savez(resolved_path, **arrays)
    return resolved_path


def _load_sdf_npz(path: Path):
    model_to_load = IGRModel(input_dim=3, depth=4, hidden=128, act=softplus, radius_init=1.0)
    with np.load(path) as data:
        flat_params = {
            tuple(key.removeprefix("params/").split("/")): jnp.array(data[key])
            for key in data.files
            if key.startswith("params/")
        }
        params = freeze(unflatten_dict(flat_params))
        normalization_params = {
            "min_coords": jnp.array(data["normalization/min_coords"]),
            "max_coord": jnp.array(data["normalization/max_coord"]),
        }

    def loaded_model_fn(x):
        return model_to_load.apply({"params": params}, x).squeeze()

    print(f"Restored portable SDF checkpoint from {path}")
    return loaded_model_fn, normalization_params


def load_sdf_model(ckpt_dir=None):
    """Load the trained IGR SDF model and normalization parameters."""
    resolved_ckpt_dir = _checkpoint_dir(ckpt_dir)
    portable_path = _portable_checkpoint_path(resolved_ckpt_dir)
    if portable_path.exists():
        return _load_sdf_npz(portable_path)

    if not os.environ.get("TSVEC_ALLOW_ORBAX_RESTORE"):
        raise FileNotFoundError(
            f"No portable SDF checkpoint found at {portable_path}. "
            "Run `python scripts/train_sdf.py` first. If you intentionally want "
            "to try a legacy Orbax checkpoint, set TSVEC_ALLOW_ORBAX_RESTORE=1."
        )

    checkpoint_data = checkpoints.restore_checkpoint(str(resolved_ckpt_dir), target=None)
    if checkpoint_data is None:
        raise FileNotFoundError(
            f"No SDF checkpoint found at {resolved_ckpt_dir}. "
            "Run `python scripts/train_sdf.py` first or set TSVEC_SDF_CHECKPOINT."
        )

    model_to_load = IGRModel(input_dim=3, depth=4, hidden=128, act=softplus, radius_init=1.0)
    loaded_model_state = checkpoint_data["sdf_state"]
    normalization_params = checkpoint_data["normalization_params"]

    if hasattr(loaded_model_state, "params"):
        def loaded_model_fn(x):
            return model_to_load.apply(
                {"params": loaded_model_state.params, **loaded_model_state.model_state}, x
            ).squeeze()
    else:
        def loaded_model_fn(x):
            return model_to_load.apply(
                {"params": loaded_model_state["params"], **loaded_model_state["model_state"]}, x
            ).squeeze()

    print(f"Restored SDF checkpoint from {resolved_ckpt_dir}")
    return loaded_model_fn, normalization_params


loaded_model_fn = None
normalization_params = None


def load_default_sdf_model():
    """Load the default checkpoint into module-level compatibility variables."""
    global loaded_model_fn, normalization_params
    loaded_model_fn, normalization_params = load_sdf_model()
    return loaded_model_fn, normalization_params
