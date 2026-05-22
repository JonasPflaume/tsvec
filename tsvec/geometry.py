import jax.numpy as jnp
import jax
import jaxlie
import numpy as np

""" compute the Gauss Newton SE3 Jacobian
    by hand 
"""

def build_pos_Jacobian(horizon:int, dim:int):
    ''' build the constant position jacobian
    '''
    J = np.zeros([horizon-2, horizon])
    lower_idx = np.arange(horizon-2)
    lower_idx = (lower_idx, lower_idx)

    upper_idx = np.arange(horizon-2)
    upper_idx = (upper_idx, upper_idx+2)

    diag_idx = np.arange(horizon-2)
    diag_idx = (diag_idx, diag_idx+1)

    J[lower_idx] = 1.0
    J[upper_idx] = 1.0
    J[diag_idx] = -2.0
    
    J = np.kron(J, np.eye(dim))
    return J

def block_jacobian(pair_jac_inv:jax.Array):
    """
    """
    t_1_block = pair_jac_inv[0]
    t_block = pair_jac_inv[1]
    H1 = t_1_block.T
    H2 = - t_1_block - t_block.T
    H3 = t_block
    return jnp.concatenate([H1, H2, H3], axis=-1)


def scatter_shifted(tensor, row_flat, col_flat):
    """
    tensor: (N, H, W)
    returns: container (N*H, N*H + W - H)
    """
    N, H, W = tensor.shape
    out_shape = (N*H, N*H+W-H)

    # batch_idx = jnp.arange(N)[:, None, None]  # (N, 1, 1)
    # row_idx = jnp.arange(H)[None, :, None]    # (1, H, 1)
    # col_idx = jnp.arange(W)[None, None, :]    # (1, 1, W)

    # global_row = batch_idx * H + row_idx      # (N, H, 1)
    # global_col = batch_idx * H + col_idx      # (N, 1, W)

    # global_row = jnp.broadcast_to(global_row, (N, H, W))  # (N, H, W)
    # global_col = jnp.broadcast_to(global_col, (N, H, W))  # (N, H, W)
    
    # row_flat = global_row.flatten()  # (N*H*W,)
    # col_flat = global_col.flatten()  # (N*H*W,)
    val_flat = tensor.flatten()      # (N*H*W,)
    
    container = jnp.zeros(out_shape, dtype=tensor.dtype)
    return container.at[row_flat, col_flat].set(val_flat)

# @profile
def build_so3_Jacobian(batch_so3_diff:jaxlie.SO3, row_flat:jax.Array, col_flat:jax.Array):
    """ single traj jacobian rot
    """
    batch_left_jac_inv = batch_so3_diff.jlog()
    early = batch_left_jac_inv[:-1][:,None,...]
    later = batch_left_jac_inv[1:][:,None,...]
    pair_jac_inv = jnp.concatenate([early, later], axis=1)
    block_jac = jax.vmap(block_jacobian) (pair_jac_inv)
    J = scatter_shifted(block_jac, row_flat, col_flat)
    return J


