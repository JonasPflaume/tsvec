
import jax
import jax.numpy as jnp
import jaxkd as jk
from tsvec.param import Param
from tsvec.planner import config

def get_metrics(state, problem):
    """
    """
    pos_particle = state.pos_particles
    rot_particle = state.rot_particles
    all_pos = pos_particle.reshape(-1, 3)  # (particle_num * horizon, 3)
    all_nids, _ = jk.query_neighbors(problem.ergodic_problem.pcd_tree, all_pos, 
                                    k=problem.ergodic_problem.nb_max_neighbors, cuda=False)
    batch_nids = all_nids.reshape(1, state.horizon, problem.ergodic_problem.nb_max_neighbors)
        
    def residual_with_precomputed_knn(pos_particle, rot_particle, boundary_points, kappa, v0, nids):
        return problem.residual_with_nids(pos_particle, rot_particle, boundary_points, kappa, v0, nids)
    
    batch_residual = jax.vmap(residual_with_precomputed_knn, in_axes=[0,0,None,0,0,0])
    res, _, _, _ = batch_residual(pos_particle, rot_particle,
                                    state.initial_points, state.kappa, state.v0, batch_nids)
    param = Param()
    Va = jnp.mean(res, axis=0)[param.nb_eigen+6*config['horizon']:param.nb_eigen+6*config['horizon']+config['horizon']]
    Vs = jnp.mean(res, axis=0)[param.nb_eigen:param.nb_eigen+6*config['horizon']]
    Vf = jnp.mean(res, axis=0)[param.nb_eigen+6*config['horizon']+config['horizon']:param.nb_eigen+6*config['horizon']+config['horizon']+config['horizon']]
    Ve = jnp.mean(res, axis=0)[:param.nb_eigen]
    
    assert len(Va) + len(Vs) + len(Vf) + len(Ve) == res.shape[-1]
    
    Va = 0.5 * jnp.sum(Va**2.0)
    Vs = 0.5 * jnp.sum(Vs**2.0)
    Vf = 0.5 * jnp.sum(Vf**2.0)
    Ve = 0.5 * jnp.sum(Ve**2.0)
    V = Va + Vs + Vf + Ve

    return {"Vs": Vs.item(), "Va": Va.item(), "Vf": Vf.item(), "Ve": Ve.item(), "V": V.item()}
