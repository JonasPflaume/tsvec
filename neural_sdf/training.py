import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import functools as ft
import optax
from neural_sdf.sdf_jax.util import plot3d, plot2d
from neural_sdf.models import IGRModel
import trimesh
from typing import Callable
from flax.training import train_state
from tqdm import tqdm
import flax
from neural_sdf.sdf_jax.util import dataloader
from neural_sdf.models import softplus
from flax.training import checkpoints
import os
from pathlib import Path

jax.config.update("jax_default_matmul_precision", "float32")

def load_mesh(path: str, target_faces: int = 50000, unit:str="m", diagnostics: bool = False):
    """Loads a triangle mesh and samples faces based on area importance. Rescales to fit unit cube [0,1]^3."""
    mesh = trimesh.load(path)
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    else:
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    vertices = jnp.array(mesh.vertices)
    faces = jnp.array(mesh.faces)
    if unit == "mm":
        vertices /= 1000.0  # convert mm to m
    
    # Calculate face areas for importance sampling
    triangles = vertices[faces]  # Shape: (n_faces, 3, 3)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    
    # Calculate area using cross product
    cross_product = jnp.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * jnp.linalg.norm(cross_product, axis=1)
    
    # Normalize areas to get sampling probabilities (larger area = higher probability)
    area_probs = face_areas / jnp.sum(face_areas)
    
    # Sample faces based on area importance
    key = jrandom.PRNGKey(42)
    selected_face_indices = jrandom.choice(
        key, 
        faces.shape[0], 
        shape=(min(target_faces, faces.shape[0]),), 
        p=area_probs, 
        replace=False
    )
    
    # Get selected faces
    selected_faces = faces[selected_face_indices]
    
    # Find all unique vertices that are used in selected faces
    unique_vertex_indices = jnp.unique(selected_faces.flatten())
    
    # Create mapping from old vertex indices to new vertex indices
    vertex_mapping = jnp.zeros(vertices.shape[0], dtype=jnp.int32) - 1  # Initialize with -1
    vertex_mapping = vertex_mapping.at[unique_vertex_indices].set(jnp.arange(len(unique_vertex_indices)))
    
    # Get the vertices that are actually used
    selected_vertices = vertices[unique_vertex_indices]
    
    # Remap face indices to the new vertex indices
    remapped_faces = vertex_mapping[selected_faces]
    selected_areas = face_areas[selected_face_indices]
    
    if diagnostics:
        # Visualization comparison
        sub_vertices_orig = vertices[::max(1, len(vertices)//1000)]
        sub_vertices_selected = selected_vertices[::max(1, len(selected_vertices)//1000)]
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 5))
        
        # Original vertices
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(sub_vertices_orig[:, 0], sub_vertices_orig[:, 1], sub_vertices_orig[:, 2], s=1, alpha=0.6)
        ax1.scatter(0.,0.,0., s=10, c='r')
        ax1.set_title(f'Original Vertices ({len(vertices)})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_aspect('equal')
        
        # Selected vertices
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(sub_vertices_selected[:, 0], sub_vertices_selected[:, 1], sub_vertices_selected[:, 2], s=1, alpha=0.6, c='orange')
        ax2.scatter(0.,0.,0., s=10, c='r')
        ax2.set_title(f'Area-importance Sampled Vertices ({len(selected_vertices)})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_aspect('equal')
        
        # Face areas histogram
        ax3 = fig.add_subplot(133)
        ax3.hist(face_areas, bins=50, alpha=0.5, label='All faces', edgecolor='black')
        ax3.hist(selected_areas, bins=50, alpha=0.7, label='Selected faces', edgecolor='red', color='orange')
        ax3.set_xlabel('Face Area')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Face Areas Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
    
    # Normalize the selected vertices to unit cube
    min_coords = selected_vertices.min(axis=0)
    print(f"Min coords: {min_coords}")
    selected_vertices -= min_coords
    max_coord = selected_vertices.max()
    print(f"Max coord: {max_coord}")
    selected_vertices /= max_coord
    
    # Store normalization parameters
    normalization_params = {
        'min_coords': min_coords,
        'max_coord': max_coord
    }
    
    print(f"Original: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Selected: {len(selected_vertices)} vertices, {len(remapped_faces)} faces")
    print(f"Selected face areas - min: {selected_areas.min():.6f}, max: {selected_areas.max():.6f}, mean: {selected_areas.mean():.6f}")
    print(f"Original face areas - min: {face_areas.min():.6f}, max: {face_areas.max():.6f}, mean: {face_areas.mean():.6f}")
    
    return selected_vertices, remapped_faces, normalization_params

# --- Loss Functions (Unchanged, but model call will be different) ---

def sample_normal_per_point(key, xs, local_sigma=0.01):
    key, key_local, key_global = jrandom.split(key, 3)
    sample_local = xs + jrandom.normal(key_local, xs.shape) * local_sigma
    sample_global = jrandom.uniform(key_global, (xs.shape[0]//8, xs.shape[1]))
    return jnp.vstack([sample_local, sample_global])

def surface_loss_fn(model_apply, params, x):
    return jnp.abs(model_apply(params, x))

def normal_loss_fn(model_apply, params, x, normal):
    # Flax model's apply function needs to be passed to jax.grad
    grad_fn = jax.grad(lambda p, x_arg: model_apply(p, x_arg).squeeze(), argnums=1)
    return jnp.linalg.norm(grad_fn(params, x) - normal)

def eikonal_loss_fn(model_apply, params, x):
    grad_fn = jax.grad(lambda p, x_arg: model_apply(p, x_arg).squeeze(), argnums=1)
    return (jnp.linalg.norm(grad_fn(params, x)) - 1.0)**2

@ft.partial(jax.value_and_grad, has_aux=True)
def loss_fn(params: flax.core.FrozenDict, state: flax.core.FrozenDict, model_apply: Callable, xs, normals, lam, tau, key):
    # Combine params and state for model application
    variables = {'params': params, **state}
    
    # Note: model_apply is now passed explicitly
    # We partially apply the model_apply function and the variables (params)
    # The vmapped function will only take 'x' or 'x' and 'normal' as input
    surface_loss = jnp.mean(jax.vmap(lambda x: surface_loss_fn(model_apply, variables, x))(xs))
    
    if normals is not None:
        normal_loss = jnp.mean(jax.vmap(lambda x, normal: normal_loss_fn(model_apply, variables, x, normal))(xs, normals))
    else:
        normal_loss = 0.0
        
    xs_eik = sample_normal_per_point(key, xs)
    eikonal_loss = jnp.mean(jax.vmap(lambda x: eikonal_loss_fn(model_apply, variables, x))(xs_eik))
    
    loss = surface_loss + tau * normal_loss + lam * eikonal_loss
    # aux_output is empty as we don't need to return an updated model instance
    return loss, {}

# Use a Flax TrainState for convenience
class TrainState(train_state.TrainState):
    model_state: flax.core.FrozenDict

@jax.jit
def train_step(state: TrainState, xs, normals, lam, tau, key):
    (loss, _), grads = loss_fn(state.params, state.model_state, state.apply_fn, xs, normals, lam, tau, key)
    state = state.apply_gradients(grads=grads)
    return loss, state

def print_callback(step, loss, state: TrainState):
    print(f"[{step}] loss: {loss:.8f}")

def fit(
    xs,
    normals=None,
    lam=1.0,
    tau=1.0,
    # module
    depth=4,
    hidden=64,
    act=softplus,
    radius_init=1.0,
    # optimizer
    key=jrandom.PRNGKey(1234),
    lr=2e-3,
    steps=100,
    batch_size=128,
    # utils
    cb=print_callback,
    cb_every=100,
):
    key, model_key = jrandom.split(key, 2)
    model = IGRModel(
        input_dim=3, depth=depth, hidden=hidden, act=act, radius_init=radius_init,
    )
    variables = model.init(model_key, xs[0])
    
    # Create Flax TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        model_state={k: v for k, v in variables.items() if k != 'params'},
        tx=optax.adam(lr)
    )

    key, data_key = jrandom.split(key, 2)
    for step, (xs_batch, normals_batch) in zip(tqdm(range(steps)), dataloader(xs, normals, batch_size, key=data_key)):
        key, step_key = jrandom.split(key, 2)
        loss, state = train_step(state, xs_batch, normals_batch, lam, tau, step_key)
        if step % cb_every == 0:
            cb(step, loss, state)
    cb(step, loss, state)
    
    # Return the final loss and a callable model function for inference/plotting
    # final_model_fn = lambda x: state.apply_fn({'params': state.params, **state.model_state}, x)
    return loss, state

def normalize(v):
    return v / jnp.linalg.norm(v)

def triangle_normal(triangle):
    v1, v2, v3 = triangle
    return normalize(jnp.cross(v2 - v1, v3 - v1))

def vertex_normals(vertices, faces):
    """Computes vertex normals by uniform averaging of triangle normals."""
    triangles = vertices[faces]
    trinormals = jax.vmap(triangle_normal)(triangles)

    normals = jnp.zeros_like(vertices)
    normals = normals.at[faces[:,0]].add(trinormals)
    normals = normals.at[faces[:,1]].add(trinormals)
    normals = normals.at[faces[:,2]].add(trinormals)
    return jax.vmap(normalize)(normals)

if __name__ == "__main__":
    release_root = Path(__file__).resolve().parents[1]
    vertices, faces, normalization_params = load_mesh(
        str(release_root / "ergodic_dataset" / "torus" / "torus.obj"), target_faces=500000
        # "ergodic_dataset/cylinder/c4.obj", target_faces=50000,
    )
    
    xs = vertices
    normals = vertex_normals(vertices, faces)
    
    # Train SDF model
    print("Starting SDF training...")
    loss, sdf_state = fit(
        xs, normals=normals,
        lam=0.1, tau=1.0,
        depth=4, hidden=128,
        lr=5e-5, steps=12000,
        batch_size=1024,
    )
    
    # Create model function for inference
    sdf_model_fn = lambda x: sdf_state.apply_fn(
        {'params': sdf_state.params, **sdf_state.model_state}, x
    ).squeeze()
    
    print("SDF model training completed successfully!")
    
    # Save checkpoint (including both sdf_state and normalization_params)
    checkpoint_dir = release_root / "checkpoints" / "sdf_model"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    from neural_sdf.checkpoint import PORTABLE_CHECKPOINT_NAME, save_sdf_npz

    portable_checkpoint = save_sdf_npz(
        checkpoint_dir / PORTABLE_CHECKPOINT_NAME,
        sdf_state,
        normalization_params,
    )
    print(f"Portable checkpoint saved to: {portable_checkpoint}")
    print(f"Saved normalization params: {normalization_params}")
    
    plot3d(sdf_model_fn, ngrid=50)
    
    
