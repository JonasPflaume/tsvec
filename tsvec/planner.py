### 
# TSVec trajectory optimization
# Jiayun Li, Aug. 17 2025
###

import jax
import jax.numpy as jnp
from jaxlie import SO3, manifold
from flax import struct
from tsvec.geometry import build_pos_Jacobian, build_so3_Jacobian
from tsvec.ergodic import ErgodicProblem
import jaxkd as jk
from tsvec.backend import use_jaxkd_cuda


config = {
    "init_lr": 1.0,
    "max_inner_inter": 1000,
    "pho_v": 1.2,
    "smoothness_weight": 5.0,
    "ergodic_weight": 0.1,
    "h_weight": 3.,
    "init_ker_l": 0.5, 
    "h_values_threshold": 1e-6, # this is f_tol
    "horizon": 200,
    "particle_num": 1,
    "ls_improve": 1.2,
    "ls_decrease": 0.5,
    "max_ls_iter": 5,
    "init_noise": 0.01,
}


@struct.dataclass
class TSVecState:
    
    rot_particles: SO3
    pos_particles: jax.Array
    kappa: jax.Array # equality penalty multiplier
    
    curr_logll: jax.Array
    curr_h_values: jax.Array
    horizon: int = struct.field(pytree_node=False)
    particle_num: int = struct.field(pytree_node=False)
    
    pho_v: float = struct.field(pytree_node=False)
    v0: jax.Array
    
    initial_points: jax.Array
    init_lr: float
    
    median_kernel: float
    
    max_inner_inter: int = struct.field(pytree_node=False)
    h_values_threshold: float = struct.field(pytree_node=False)
    
    # Add new fields for line search
    armijo_c1: float = struct.field(pytree_node=False)
    ls_decrease: float = struct.field(pytree_node=False)
    ls_improve: float = struct.field(pytree_node=False)
    max_ls_iter: int = struct.field(pytree_node=False)
    damping_factor: float = struct.field(default=1e-3) # single precision 
    
    @classmethod
    def create(cls, 
               init_points:jax.Array, 
               config: dict
               ):
        """ """
        particle_num = config.get("particle_num", 1)
        horizon = config.get("horizon", 40)
        pho_v = config.get("pho_v", 1.2)
        init_lr = config.get("init_lr", 1.0)
        max_inner_inter = config.get("max_inner_inter", 100)

        batch_axes = (particle_num, horizon)
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        random_noise_pos = jax.random.normal(key1, (*batch_axes, 3)) * config["init_noise"]
        random_noise_quat = jax.random.normal(key2, (*batch_axes, 4)) * config["init_noise"]
        linspace_traj = (init_points[1] - init_points[0]) / (horizon - 1)
        t = jnp.arange(horizon)
        traj = init_points[0] + linspace_traj * t[:, None]
        traj = jnp.broadcast_to(traj, (*batch_axes, 3))
        traj = traj.at[:,1:-1,:].add(random_noise_pos[:,1:-1,:])
        
        quat = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (*batch_axes, 4))
        quat = quat + random_noise_quat
        quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
        
        h_values = jnp.ones((particle_num, horizon)) * 1e8
        k = jnp.zeros((particle_num, 2*horizon+6)) # equality penalty multiplier
        logll = jnp.ones((particle_num,)) * -1e8 
        
        return cls(
            rot_particles=SO3(wxyz=quat),
            pos_particles=traj,
            kappa=k,
            curr_logll=logll,
            curr_h_values=h_values,
            horizon=horizon,
            particle_num=particle_num,
            pho_v=pho_v,
            v0=config.get("h_weight", 1.0) * jnp.ones((particle_num, )),
            initial_points=init_points,
            init_lr=init_lr,
            max_inner_inter=max_inner_inter,
            median_kernel=config.get("init_ker_l", 0.05),
            h_values_threshold=config.get("h_values_threshold", 5e-5),
            armijo_c1=config.get("armijo_c1", 1e-4),
            ls_decrease=config.get("ls_decrease", 0.5),
            ls_improve=config.get("ls_improve", 1.2),
            max_ls_iter=config.get("max_ls_iter", 10)
        )
    
@struct.dataclass
class TSVecProblem:
    smoothness_weight: float = struct.field(pytree_node=False)
    ergodic_weight: float = struct.field(pytree_node=False)
    sdf_fn: callable = struct.field(pytree_node=False)
    
    Jpos: jax.Array# = struct.field(pytree_node=False)
    JTJpos: jax.Array# = struct.field(pytree_node=False)
    
    Jrot_sparse_row_index: jax.Array# = struct.field(pytree_node=False)
    Jrot_sparse_col_index: jax.Array# = struct.field(pytree_node=False)
    
    Hess_block_row_index: jax.Array# = struct.field(pytree_node=False)
    Hess_block_col_index: jax.Array# = struct.field(pytree_node=False)
    
    ergodic_problem: ErgodicProblem# = struct.field(pytree_node=False)
    
    @classmethod
    def create(cls, config: dict, sdf_fn: callable, ergodic_problem: ErgodicProblem):
        J = build_pos_Jacobian(config['horizon'], 3) * jnp.sqrt(config['smoothness_weight'])
        JTJ = J.T @ J
        
        # sparse Jrot structure
        N = config['horizon'] - 2
        H = 3
        W = H * 3
        batch_idx = jnp.arange(N)[:, None, None]  # (N, 1, 1)
        row_idx = jnp.arange(H)[None, :, None]    # (1, H, 1)
        col_idx = jnp.arange(W)[None, None, :]    # (1, 1, W)

        global_row = batch_idx * H + row_idx      # (N, H, 1)
        global_col = batch_idx * H + col_idx      # (N, 1, W)

        global_row = jnp.broadcast_to(global_row, (N, H, W))  # (N, H, W)
        global_col = jnp.broadcast_to(global_col, (N, H, W))  # (N, H, W)
        
        row_flat = global_row.flatten()  # (N*H*W,)
        col_flat = global_col.flatten()  # (N*H*W,)
        
        # Build the diagonal hessian index, (H, 3, 3) add to big hessian (3*H, 3*H)
        horizon = config['horizon']
        block_size = 3
        block_offset = jnp.arange(horizon) * block_size  # (horizon,)

        # Row/Col offsets within each block
        local_r = jnp.arange(block_size)  # (block_size,)
        local_c = jnp.arange(block_size)  # (block_size,)

        # Get (r, c) pairs within each block
        rr, cc = jnp.meshgrid(local_r, local_c, indexing="ij")  # shape (block_size, block_size)
        rr = rr.flatten()  # (block_size^2,)
        cc = cc.flatten()

        # Broadcast to all blocks
        rr_all = block_offset[:, None] + rr[None, :]  # (horizon, block_size^2)
        cc_all = block_offset[:, None] + cc[None, :]

        hess_row, hess_col = rr_all.flatten(), cc_all.flatten()
        
        return cls(smoothness_weight=config['smoothness_weight'], 
                   ergodic_weight=config['ergodic_weight'],
                   sdf_fn=sdf_fn, Jpos=J, JTJpos=JTJ,
                   Jrot_sparse_row_index=row_flat, Jrot_sparse_col_index=col_flat,
                   Hess_block_row_index=hess_row, Hess_block_col_index=hess_col,
                     ergodic_problem=ergodic_problem
                   )
        
    def compute_smoothness(self, pos_particle: jax.Array, rot_particle: SO3):
        """
        """
        # pos_particle shape: (N, H, 3), rot_particle shape: (N, H)
        # pos_residual = jnp.diff(jnp.diff(pos_particle, axis=-2), axis=-2).reshape(-1)
        pos_residual = pos_particle[...,2:,:] - 2 * pos_particle[...,1:-1,:] + pos_particle[...,:-2,:]
        pos_residual = pos_residual.reshape(-1)

        t_1 = SO3(rot_particle.wxyz[:-1,...])
        t_2 = SO3(rot_particle.wxyz[1:,...])
        so3_diff = t_1.inverse() @ t_2
        so3_diff_log = so3_diff.log()
        rot_residual = (so3_diff_log[...,1:,:] - so3_diff_log[...,:-1,:]).reshape(-1)

        rot_J = build_so3_Jacobian(so3_diff, self.Jrot_sparse_row_index, self.Jrot_sparse_col_index)
        return pos_residual, rot_residual, rot_J
    
    def residual(self, pos_particle: jax.Array, rot_particle: SO3, 
                 boundary_points: jax.Array, kappa: jax.Array, v0: float, ergodic_problem:ErgodicProblem):
        """ 
        Planner residual
        return the residual, gradient and the GN hessian
        """
        # compute the ergodic residual
        ergo_res, ergo_J = ergodic_problem.ergodic_residual(pos_particle)
        ergo_res = jnp.sqrt(self.ergodic_weight) * ergo_res
        ergo_J = jnp.sqrt(self.ergodic_weight) * ergo_J.reshape(ergo_J.shape[0], -1)

        # compute the residual
        sqrt_smoothness_weight = jnp.sqrt(self.smoothness_weight)
        pos_residual, rot_residual, rot_J = self.compute_smoothness(pos_particle, rot_particle)
        pos_residual = sqrt_smoothness_weight * pos_residual
        rot_residual = sqrt_smoothness_weight * rot_residual
        rot_J = sqrt_smoothness_weight * rot_J
        # compute the h_residual

        h1_residual, h2_residual, h1_p_J, h1_R_J, h2_p_J =\
                                self.equality(pos_particle, rot_particle)
        sqrt_v0 = jnp.sqrt(v0)
        h1_residual = sqrt_v0 * h1_residual
        h2_residual = sqrt_v0 * h2_residual
        h1_p_J = sqrt_v0 * h1_p_J
        h1_R_J = sqrt_v0 * h1_R_J
        h2_p_J = sqrt_v0 * h2_p_J

        # r1, r2, b1_J, b2_J = self.boundary(pos_particle, boundary_points)
        # r1 = sqrt_v0 * r1
        # r2 = sqrt_v0 * r2
        # b1_J = sqrt_v0 * b1_J
        # b2_J = sqrt_v0 * b2_J
        ### compute all residuals
        total_residual = jnp.concatenate([ergo_res, pos_residual, rot_residual, 
                                          h1_residual, h2_residual], axis=0)
        
        ### position gradient 
        g_pos =     self.Jpos.T @ pos_residual + ergo_J.T @ ergo_res + \
                    (h1_p_J * h1_residual[:,None]).reshape(-1) + \
                    (h2_p_J * h2_residual[:,None]).reshape(-1) + \
                    (h1_p_J * kappa[:h1_p_J.shape[0],None]).reshape(-1) +\
                    (h2_p_J * kappa[h1_p_J.shape[0]:h1_p_J.shape[0]+h2_p_J.shape[0],None]).reshape(-1)

        # leading_g_pos = b1_J.T @ r1 + b1_J.T @ kappa[-6:-3]
        # ending_g_pos = b2_J.T @ r2 + b2_J.T @ kappa[-3:]
        # g_pos = g_pos.at[0:3].add(leading_g_pos).at[-3:].add(ending_g_pos)
        
        ### rotation gradient
        g_rot = rot_J.T @ rot_residual + (h1_R_J * h1_residual[:,None]).reshape(-1) +\
                (h1_R_J * kappa[h1_p_J.shape[0]:h1_p_J.shape[0]+h2_p_J.shape[0]:,None]).reshape(-1)
        
        
        Hpp_diag =  (
                        jnp.einsum("ni,nj->nij", h1_p_J, h1_p_J) +
                        jnp.einsum("ni,nj->nij", h2_p_J, h2_p_J)
                    ).reshape(-1) 
        Hpp = (self.JTJpos + ergo_J.T @ ergo_J).at[self.Hess_block_row_index, 
                                                   self.Hess_block_col_index].add(Hpp_diag)\
                        #.at[0:3, 0:3].add(b1_J.T @ b1_J).at[-3:,-3:].add(b2_J.T @ b2_J)
        
        Hrr_diag = jnp.einsum("ni,nj->nij", h1_R_J, h1_R_J).reshape(-1)       
        Hrr = (rot_J.T @ rot_J).at[self.Hess_block_row_index, self.Hess_block_col_index].add(Hrr_diag)

        Hpr_diag = jnp.einsum("ni,nj->nij", h1_p_J, h1_R_J).reshape(-1) 
        Hpr = jnp.zeros((Hpp.shape[0], Hrr.shape[1]))\
                .at[self.Hess_block_row_index, self.Hess_block_col_index].set(Hpr_diag)
        Hrp = Hpr.transpose(1, 0)
        H_GN = jnp.block([
            [Hpp, Hpr],
            [Hrp, Hrr]
        ])
        return total_residual, g_pos, g_rot, H_GN
    
    def residual_with_nids(self, pos_particle: jax.Array, rot_particle: SO3, 
                      boundary_points: jax.Array, kappa: jax.Array, v0: float, nids: jax.Array):
        """ 
        Planner residual with precomputed KNN indices
        """

        ergo_res, ergo_J = self.ergodic_problem.ergodic_residual_with_nids(pos_particle, nids)
        ergo_res = jnp.sqrt(self.ergodic_weight) * ergo_res
        ergo_J = jnp.sqrt(self.ergodic_weight) * ergo_J.reshape(ergo_J.shape[0], -1)

        # compute the residual
        sqrt_smoothness_weight = jnp.sqrt(self.smoothness_weight)
        pos_residual, rot_residual, rot_J = self.compute_smoothness(pos_particle, rot_particle)
        pos_residual = sqrt_smoothness_weight * pos_residual
        rot_residual = sqrt_smoothness_weight * rot_residual
        rot_J = sqrt_smoothness_weight * rot_J
        # compute the h_residual

        h1_residual, h2_residual, h1_p_J, h1_R_J, h2_p_J =\
                                self.equality(pos_particle, rot_particle)
        sqrt_v0 = jnp.sqrt(v0)
        h1_residual = sqrt_v0 * h1_residual
        h2_residual = sqrt_v0 * h2_residual
        h1_p_J = sqrt_v0 * h1_p_J
        h1_R_J = sqrt_v0 * h1_R_J
        h2_p_J = sqrt_v0 * h2_p_J

        # ...existing code... (rest same as original residual method)
        ### compute all residuals
        total_residual = jnp.concatenate([ergo_res, pos_residual, rot_residual, 
                                        h1_residual, h2_residual], axis=0)
        
        ### position gradient 
        g_pos =     self.Jpos.T @ pos_residual + ergo_J.T @ ergo_res + \
                    (h1_p_J * h1_residual[:,None]).reshape(-1) + \
                    (h2_p_J * h2_residual[:,None]).reshape(-1) + \
                    (h1_p_J * kappa[:h1_p_J.shape[0],None]).reshape(-1) +\
                    (h2_p_J * kappa[h1_p_J.shape[0]:h1_p_J.shape[0]+h2_p_J.shape[0],None]).reshape(-1)

        ### rotation gradient
        g_rot = rot_J.T @ rot_residual + (h1_R_J * h1_residual[:,None]).reshape(-1) +\
                (h1_R_J * kappa[h1_p_J.shape[0]:h1_p_J.shape[0]+h2_p_J.shape[0]:,None]).reshape(-1)
        
        Hpp_diag =  (
                        jnp.einsum("ni,nj->nij", h1_p_J, h1_p_J) +
                        jnp.einsum("ni,nj->nij", h2_p_J, h2_p_J)
                    ).reshape(-1) 
        Hpp = (self.JTJpos + ergo_J.T @ ergo_J).at[self.Hess_block_row_index, 
                                                self.Hess_block_col_index].add(Hpp_diag)
        
        Hrr_diag = jnp.einsum("ni,nj->nij", h1_R_J, h1_R_J).reshape(-1)       
        Hrr = (rot_J.T @ rot_J).at[self.Hess_block_row_index, self.Hess_block_col_index].add(Hrr_diag)

        Hpr_diag = jnp.einsum("ni,nj->nij", h1_p_J, h1_R_J).reshape(-1) 
        Hpr = jnp.zeros((Hpp.shape[0], Hrr.shape[1]))\
                .at[self.Hess_block_row_index, self.Hess_block_col_index].set(Hpr_diag)
        Hrp = Hpr.transpose(1, 0)
        H_GN = jnp.block([
            [Hpp, Hpr],
            [Hrp, Hrr]
        ])
        return total_residual, g_pos, g_rot, H_GN
    
    def boundary(self, pos_particle: jax.Array, boundary_points: jax.Array):
        """
        """
        r1 = pos_particle[0] - boundary_points[0]
        r2 = pos_particle[-1] - boundary_points[1]
        b1_J = jnp.eye(3)
        b2_J = jnp.eye(3)
        return r1, r2, b1_J, b2_J
        
    def equality(self, pos_particle: jax.Array, rot_particle: SO3):
        """ use 1e-1 to scale the h1 alignment residual, to make it has 
            the same scale as the h2 residual (0.1 meter)
        """
        align_scale = 1e-1
        value, grad = jax.vmap(jax.value_and_grad(self.sdf_fn))(pos_particle)
        h2_p_J = grad
        h2_residual = value
        
        def single_h1_p_J(pos_particle: jax.Array, rot_particle: SO3):
            v = rot_particle.apply(jnp.array([0.0, 0.0, 1.0]))
            return jax.jvp(jax.grad(self.sdf_fn), (pos_particle,), (v,))[1]
        
        def single_h1_R_J(rot_particle: SO3, grad_t: jax.Array):
            return jnp.cross(jnp.array([0.0, 0.0, 1.0]), rot_particle.inverse().apply(grad_t)).T
        ### here is where to flip the Z axis, set to -1 to outwards, 1 for inwards
        h1_residual = jnp.einsum("bi,bi->b", rot_particle.apply(jnp.array([0.0, 0.0, 1.0])), grad) - 1.0
        h1_p_J = jax.vmap(single_h1_p_J, in_axes=(0, 0))(pos_particle, rot_particle)
        h1_R_J = jax.vmap(single_h1_R_J, in_axes=(0, 0))(rot_particle, grad)

        return (align_scale*h1_residual, h2_residual, 
                align_scale*h1_p_J, align_scale*h1_R_J, h2_p_J)

class TSVecDirection:
    
    @staticmethod
    def compute(pos_particles, rot_particles, pos_particles2, rot_particles2, 
               pos_grad, rot_grad, MetricH, l):
        '''
        '''
        par_num, horizon, _ = pos_particles.shape

        xi_pos = pos_particles[:, None, ...] - pos_particles2[None, :, ...]
        rot_particle_1_gram = SO3(rot_particles.wxyz[:, None, ...])
        rot_particle_2_gram = SO3(rot_particles2.wxyz[None, :, ...])
        xi_rot_so3 = rot_particle_1_gram.inverse() @ rot_particle_2_gram
        xi_rot = xi_rot_so3.log()
        # compute the gram matrix
        gram_pos = jnp.exp(-jnp.sum(xi_pos.reshape(par_num, par_num, -1) ** 2.0, axis=-1) / l)
        gram_rot = jnp.exp(-jnp.sum(xi_rot.reshape(par_num, par_num, -1) ** 2.0, axis=-1) / l)
        gram = 0.5 * (gram_pos + gram_rot)

        # compute the grad kernel
        gram_pos_grad = gram_pos[...,None] * (-1.0 / l * xi_pos.reshape(par_num, par_num, -1))
        _gram_rot_grad = gram_rot[...,None,None] * (-1.0 / l * xi_rot)
        rot_grad_gram = jnp.einsum("...j,...ij->...i",  _gram_rot_grad, -xi_rot_so3.jlog())
        rot_grad_gram = rot_grad_gram.reshape(par_num, par_num, -1)
        
        gram_adj = TSVecDirection.adjoint(rot_particles, rot_particles2)
        trans_rot_grad_gram = jnp.einsum("ijkbp,ijkp->ijkb", gram_adj, rot_grad_gram.reshape(par_num, par_num, horizon, 3))
        grad_gram = jnp.concatenate([gram_pos_grad, trans_rot_grad_gram.reshape(par_num, par_num, -1)], axis=-1)
        # compute the Stein gradient
        
        stein_pos = (jnp.einsum("ij,ik->ijk", gram, -pos_grad) + gram_pos_grad).mean(axis=0)
        stein_rot = jnp.einsum("ij,ik->ijk", gram, -rot_grad)
        
        stein_rot = jnp.einsum("ijkbp,ijkp->ijkb", gram_adj, stein_rot.reshape(par_num, par_num, horizon, 3))
        stein_rot = (stein_rot + trans_rot_grad_gram).mean(axis=0)
        stein_rot = stein_rot.reshape(par_num, horizon*3)
        
        stein_gradient = jnp.concatenate([stein_pos, stein_rot], axis=-1)
        # compute the TSVec direction
        SteinMetric = MetricH[:,None,...] * gram[...,None,None] ** 2.0 +\
                        jnp.einsum("...i,...j->...ij", grad_gram, grad_gram)
        SteinMetric = jnp.mean(SteinMetric, axis=0)
        # jnp.linalg.eigvalsh(SteinMetric, symmetrize_input=True)  # for numerical stability check
        tsvec_step = jax.scipy.linalg.solve(SteinMetric, stein_gradient, assume_a='pos')

        tsvec_pos = tsvec_step[...,:3*horizon].reshape(par_num, horizon, 3)
        tsvec_rot = tsvec_step[...,3*horizon:].reshape(par_num, horizon, 3)

        return tsvec_pos, tsvec_rot
    
    
    @staticmethod
    def adjoint(rot_particles, rot_particles2):
        '''
        '''
        def getAdjoint(x:SO3, y:SO3) -> jax.Array:
            diff =  y.inverse() @ x
            return diff.adjoint()
        gram_adjoint = jax.vmap(jax.vmap(getAdjoint,in_axes=(None, 0)),in_axes=(0, None))
        return gram_adjoint(rot_particles, rot_particles2)
    
class TSVecPlanner:
    def __init__(self, use_tsvec_direction=True):
        self.use_tsvec_direction = use_tsvec_direction

    @staticmethod
    def line_search(state: TSVecState, problem: TSVecProblem,
                             update_pos: jax.Array, update_rot: jax.Array):
        """
        Simplified backtracking line search
        """
        
        candidate_lr_1 = state.init_lr * state.ls_improve ** jnp.arange(0,state.max_ls_iter)
        candidate_lr_2 = state.init_lr * state.ls_decrease ** jnp.arange(1,state.max_ls_iter+1)
        candidate_lr = jnp.concatenate([candidate_lr_1, candidate_lr_2], axis=0)
        
        pos_particle = state.pos_particles[None,...] + candidate_lr[:,None,None,None] * update_pos
        rot_particle = manifold.rplus(state.rot_particles, candidate_lr[:,None,None,None] * update_rot)
        all_pos = pos_particle.reshape(-1, 3)  # (particle_num * horizon, 3)
        all_nids, _ = jk.query_neighbors(problem.ergodic_problem.pcd_tree, all_pos, 
                                        k=problem.ergodic_problem.nb_max_neighbors, cuda=use_jaxkd_cuda())
        batch_nids = all_nids.reshape(len(candidate_lr), state.particle_num, state.horizon, problem.ergodic_problem.nb_max_neighbors)
            
        def residual_with_precomputed_knn(pos_particle, rot_particle, boundary_points, kappa, v0, nids):
            return problem.residual_with_nids(pos_particle, rot_particle, boundary_points, kappa, v0, nids)
        
        batch_residual = jax.vmap(jax.vmap(residual_with_precomputed_knn, in_axes=[0,0,None,0,0,0]), in_axes=(0,0,None,None,None,0))
        res, _, _, _ = batch_residual(pos_particle, rot_particle,
                                        state.initial_points, state.kappa, state.v0, batch_nids)
        obj = 0.5*jnp.sum(res**2, axis=-1)
        obj = jnp.mean(obj, axis=-1)

        best_lr_idx = jnp.argmin(obj)
        final_lr = candidate_lr[best_lr_idx]
        
        new_damping = jnp.where(
            final_lr < state.init_lr,
            state.damping_factor * 1.2,
            state.damping_factor * 0.8
        )

        new_damping = jnp.clip(new_damping, 1e-4, 1e1)
        # new_damping = state.damping_factor
        return final_lr, new_damping
    
    def inner_loop(self, state: TSVecState, problem: TSVecProblem):
        """ Planner step with adaptive damping """
        
        all_pos = state.pos_particles.reshape(-1, 3)  # (particle_num * horizon, 3)
        all_nids, _ = jk.query_neighbors(problem.ergodic_problem.pcd_tree, all_pos, 
                                        k=problem.ergodic_problem.nb_max_neighbors, cuda=use_jaxkd_cuda())
        batch_nids = all_nids.reshape(state.particle_num, state.horizon, problem.ergodic_problem.nb_max_neighbors)
        
        def residual_with_precomputed_knn(pos_particle, rot_particle, boundary_points, kappa, v0, nids):
            return problem.residual_with_nids(pos_particle, rot_particle, boundary_points, kappa, v0, nids)
        
        batch_residual = jax.vmap(residual_with_precomputed_knn, in_axes=[0,0,None,0,0,0])
        res, pos_grad, rot_grad, H_GN = batch_residual(
                                        state.pos_particles, state.rot_particles,
                                        state.initial_points, state.kappa, state.v0, batch_nids
                                    )
        
        # Apply adaptive damping
        H_GN = H_GN + jnp.eye(H_GN.shape[-1]) * state.damping_factor
        
        if self.use_tsvec_direction:
            pos_tsvec, rot_tsvec = TSVecDirection.compute(state.pos_particles, state.rot_particles,
                    state.pos_particles, state.rot_particles, pos_grad, rot_grad, 
                    H_GN, state.median_kernel)
            
            update_pos = pos_tsvec.reshape(state.particle_num, state.horizon, 3)
            update_rot = rot_tsvec.reshape(state.particle_num, state.horizon, 3)
            optimal_lr = state.init_lr
            new_damping = state.damping_factor
        else:
            total_grad = jnp.concatenate([pos_grad, rot_grad], axis=-1)
            natural_grad = jax.scipy.linalg.solve(H_GN, total_grad, assume_a='pos')
            pos_update = natural_grad[:,:3*state.horizon]
            rot_update = natural_grad[:,3*state.horizon:]
            update_pos = -pos_update.reshape(state.particle_num, state.horizon, 3)
            update_rot = -rot_update.reshape(state.particle_num, state.horizon, 3)

            # Perform backtracking line search
            optimal_lr, new_damping = self.line_search(state, problem, update_pos, update_rot)
            # optimal_lr, new_damping = state.init_lr, state.damping_factor  # Disable line search for now
        
        new_pos = state.pos_particles + optimal_lr * update_pos
        new_rot = manifold.rplus(state.rot_particles, optimal_lr * update_rot)
        
        return state.replace(pos_particles=new_pos, 
                             rot_particles=new_rot,
                             init_lr=optimal_lr,
                             damping_factor=new_damping,
                             curr_logll=0.5*jnp.einsum("ij,ij->i", res, res),)
