import jaxkd as jk
import jax
import jax.numpy as jnp
from typing import Any
from flax import struct
from lbfgsb.spectral import build_spectral_model, compute_target_coefficients
from lbfgsb.param import Param
from tsvec.backend import use_jaxkd_cuda


@struct.dataclass
class ErgodicProblem:
    """
    """
    # point cloud
    V: jax.Array# = struct.field(pytree_node=False)
    pcd_tree: Any# = struct.field(pytree_node=False)
    
    # spec parameters
    Phi: jax.Array# = struct.field(pytree_node=False) 
    B: jax.Array# = struct.field(pytree_node=False) 
    u: jax.Array# = struct.field(pytree_node=False) 
    lambda_weights: jax.Array# = struct.field(pytree_node=False)
    
    # target coefficients
    c_goal: jax.Array# = struct.field(pytree_node=False)
    
    # for objective only need two params
    nb_max_neighbors: int = struct.field(pytree_node=False) 
    agent_radius: float = struct.field(pytree_node=False)
    
    # second dim index
    second_dim_indices: jax.Array# = struct.field(pytree_node=False)
    
    
    @classmethod
    def create(cls, pcloud, ext_param: Param, T:int):

        u0 = pcloud.colors[:, 1].copy()
        spec = build_spectral_model(pcloud, ext_param)
        goal_density = u0 / ((spec["M"] @ u0).sum() + 1e-12)
        beta = ext_param.spectral_mix
        c_goal = compute_target_coefficients(spec, goal_density, beta=beta)
        
        pcd_tree = jk.build_tree(pcloud.vertices)
        
        second_dim_indices = jnp.arange(T)
        second_dim_indices = second_dim_indices[:,None]
        second_dim_indices = jnp.broadcast_to(second_dim_indices,(T, ext_param.nb_max_neighbors))

        return cls(
                    V=jnp.array(pcloud.vertices),
                    pcd_tree=pcd_tree,
                    Phi=jnp.array(spec["Phi"]),
                    B=jnp.array(spec["B"]),
                    u=jnp.array(spec["u"]),
                    lambda_weights=jnp.array(spec["lambda_weights"]),
                    c_goal=jnp.array(c_goal),
                    nb_max_neighbors=ext_param.nb_max_neighbors,
                    agent_radius=ext_param.agent_radius,
                    second_dim_indices=second_dim_indices,
                )

    def ergodic_residual(self, pos_particle: jax.Array):
        """
            SMC ergodic residual
        """

        
        nids, _ = jk.query_neighbors(self.pcd_tree, pos_particle, 
                                         k=self.nb_max_neighbors, cuda=use_jaxkd_cuda())
        
        # def residual(pos_particle):
        #     Np = len(self.V)
        #     cov_raw = jnp.zeros(Np, dtype=float)
        #     knn_V = self.V[nids]
        #     diff_knnv = knn_V - pos_particle[:,None,:]
        #     dists_power = jnp.sum(diff_knnv ** 2.0, axis=-1)
        #     K = jnp.exp(-(1.0 / self.agent_radius) * dists_power)
        #     cov_raw = cov_raw.at[nids].add(K)
        #     D = jnp.sum(self.u * cov_raw) + 1e-12
        #     cov = cov_raw / D
        #     c_traj = self.B @ cov
        #     diff = c_traj - self.c_goal
        #     return diff
            
        # res = residual(pos_particle)
        # ergodic_J = jax.jacobian(residual)(pos_particle)
        
        
        # hand jacobian
        Np = len(self.V)
        cov_raw = jnp.zeros(Np, dtype=float)
        knn_V = self.V[nids]
        diff_knnv = knn_V - pos_particle[:,None,:]
        dists_power = jnp.sum(diff_knnv ** 2.0, axis=-1)
        K = jnp.exp(-(1.0 / self.agent_radius) * dists_power)
        cov_raw = cov_raw.at[nids].add(K)
        D = jnp.sum(self.u * cov_raw) + 1e-12
        cov = cov_raw / D
        c_traj = self.B @ cov
        res = c_traj - self.c_goal
        
        kernel_grad = 2.0 / self.agent_radius * diff_knnv * K[...,None]
        p_c_traj_p_cov_raw = self.B / D - jnp.outer(self.B @ cov, self.u) / D
        
        p_cov_raw_p_P = kernel_grad.transpose(1,0,2)
        selectd = p_c_traj_p_cov_raw[:,nids]
        ergodic_J = jnp.einsum("ebi,ibk->ebk", selectd, p_cov_raw_p_P)
        
        return res, ergodic_J
    
    def ergodic_residual_with_nids(self, pos_particle: jax.Array, nids: jax.Array):
        """
            SMC ergodic residual with precomputed KNN indices
            pos_particle: (horizon, 3)
            nids: (horizon, nb_max_neighbors)
        """
        
        # def residual(pos_particle):
        #     Np = len(self.V)
        #     cov_raw = jnp.zeros(Np, dtype=float)
        #     knn_V = self.V[nids]
        #     diff_knnv = knn_V - pos_particle[:,None,:]
        #     dists_power = jnp.sum(diff_knnv ** 2.0, axis=-1)
        #     K = jnp.exp(-(1.0 / self.agent_radius) * dists_power)
        #     cov_raw = cov_raw.at[nids].add(K)
        #     D = jnp.sum(self.u * cov_raw) + 1e-12
        #     cov = cov_raw / D
        #     c_traj = self.B @ cov
        #     diff = c_traj - self.c_goal
        #     return diff
            
        # res = residual(pos_particle)
        # ergodic_J = jax.jacobian(residual)(pos_particle)
        
        Np = len(self.V)
        cov_raw = jnp.zeros(Np, dtype=float)
        knn_V = self.V[nids]
        diff_knnv = knn_V - pos_particle[:,None,:]
        dists_power = jnp.sum(diff_knnv ** 2.0, axis=-1)
        # K = jnp.exp(-(1.0 / self.agent_radius) * dists_power) # too sharp for small radius
        K = 1 / (1 + (dists_power/self.agent_radius)**2.0) # this is the best
        # K = 1 / jnp.sqrt(1 + dists_power / self.agent_radius ) # has numerical issue

        cov_raw = cov_raw.at[nids].add(K)
        D = jnp.sum(self.u * cov_raw) + 1e-12
        cov = cov_raw / D
        c_traj = self.B @ cov
        res = c_traj - self.c_goal
        
        kernel_grad = 2.0 / self.agent_radius * diff_knnv * K[...,None]
        p_c_traj_p_cov_raw = self.B / D - jnp.outer(self.B @ cov, self.u) / D
        
        p_cov_raw_p_P = kernel_grad.transpose(1,0,2)
        selectd = p_c_traj_p_cov_raw[:,nids]
        ergodic_J = jnp.einsum("ebi,ibk->ebk", selectd, p_cov_raw_p_P)
        
        return res, ergodic_J
