import numpy as np
import scipy.sparse.linalg as sla
import robust_laplacian


def build_spectral_model(pcloud, param):
    V = pcloud.vertices
    nn = param.nb_eigen // 5 if param.nb_eigen >= 50 else 20
    C, M = robust_laplacian.point_cloud_laplacian(V, n_neighbors=nn)
    M = M.tocsr()

    evals, evecs = sla.eigsh(C, param.nb_eigen, M, sigma=1e-12)
    exp_vector = np.exp(-evals * pcloud.dt)
    lambda_weights = 1.0 / (1.0 + evals) ** 1.2

    Phi = evecs               # (N,K)
    B = (M @ Phi).T           # (K,N)
    N = V.shape[0]
    u = (M.T @ np.ones(N)).reshape(-1)  # (N,)

    spec = dict(
        C=C, M=M, evals=evals, evecs=evecs, Phi=Phi,
        B=B, u=u, exp_vector=exp_vector, lambda_weights=lambda_weights
    )
    return spec

def compute_target_coefficients(spec, goal_density, beta=0.0):
    # c_global = exp * [ Phi^T (M @ rho) ] = exp * (B @ rho)
    c_local = spec["B"] @ goal_density
    c_global = spec["exp_vector"] * c_local
    return (1.0 - beta) * c_global + beta * c_local
