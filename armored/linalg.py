import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax, random
from jax.scipy.linalg import block_diag


# useful math functions
@jit
def log2z(x):
    return jnp.where(x > 0, jnp.log2(x), 0.)


@jit
def exp2z(x):
    return jnp.where(x > 0, 2. ** x, 0.)


# jit compile Hessian computation step
@jit
def A_next(G, Beta):
    A = jnp.einsum('ntki,kl,ntlj->ij', G, Beta, G)
    return (A + A.T) / 2.


# jit compile Hessian computation step
@jit
def A_next_ind(G, Beta):
    A = jnp.einsum('ntki,k,ntkj->ij', G, Beta, G)
    return (A + A.T) / 2.


@jit
def A_next_diag(y_precision, G):
    return jnp.einsum("j,ntjk->k", y_precision, G ** 2)

@jit
def An_next_diag(y_precision, G):
    # G has dimensions [n, n_t, n_out, n_params]
    return jnp.einsum("j,ntjk->nk", y_precision, G ** 2)

@jit
def Ai_next_diag(y_precision, G):
    return jnp.einsum("j,tjk->k", y_precision, G ** 2)


# jit compile function to compute log of determinant of a matrix
@jit
def log_det(A):
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))
batch_log_det = jit(vmap(log_det))

# jit compile inverse Hessian computation step
@jit
def Ainv_next(G, Ainv, BetaInv):
    GAinv = G @ Ainv
    Ainv_step = GAinv.T @ jnp.linalg.inv(BetaInv + GAinv @ G.T) @ GAinv
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


# jit compile inverse Hessian computation step
@jit
def Ainv_prev(G, Ainv, BetaInv):
    GAinv = G @ Ainv
    Ainv_step = GAinv.T @ jnp.linalg.inv(GAinv @ G.T - BetaInv) @ GAinv
    Ainv_step = (Ainv_step + Ainv_step.T) / 2.
    return Ainv_step


@jit
def Ainv_next_diag(G, Ainv, BetaInv):
    # G has shape [n_out, n_params]
    GAinv = jnp.einsum("ki,i->ki", G, Ainv)
    # Yinv = jnp.linalg.inv(BetaInv + GAinv @ G.T)
    Yinv = jnp.linalg.inv(BetaInv + jnp.einsum("ki,li->kl", GAinv, G))
    # Ainv_step = GAinv.T @ Yinv @ GAinv
    Ainv_step = jnp.einsum("ki, kl, li->i", GAinv, Yinv, GAinv)
    return Ainv_step


# approximate inverse of A, where A = LL^T, Ainv = Linv^T Linv
@jit
def compute_Ainv(A):
    Linv = jnp.linalg.inv(jnp.linalg.cholesky(A))
    Ainv = Linv.T @ Linv
    return Ainv


# jit compile measurement covariance computation
@jit
def compute_yCOV(errors, G, Ainv):
    return jnp.einsum('ntk,ntl->kl', errors, errors) + jnp.einsum("ntij,jl,ntml->im", G, Ainv, G)


@jit
def trace_GGM(G, M):
    # return Trace( g @ g.T @ M) = g.T @ M @ g
    # k indexes each output
    return jnp.einsum("ntki,ntkj,ij->k", G, G, M)


@jit
def GGv(G, v):
    return jnp.einsum("ntki,i->k", G ** 2, v)


# jit compile measurement covariance computation
@jit
def compute_yCOV_ind(errors, G, Ainv):
    return jnp.einsum('ntk,ntk->k', errors, errors) + jnp.einsum("ntki,ij,ntkj->k", G, Ainv, G)


# jit compile prediction covariance computation
@jit
def compute_predCOV(BetaInv, G, Ainv):
    return BetaInv + jnp.einsum("ntij,jl,ntml->ntim", G, Ainv, G)


@jit
def compute_predCOV_diag(BetaInv, G, Ainv):
    return BetaInv + jnp.einsum("ntij,j,ntmj->ntim", G, Ainv, G)


# jit compile prediction covariance computation
@jit
def compute_searchCOV(BetaInv, G, Ainv):
    # dimensions of samples
    n, n_t, n_y, n_theta = G.shape
    # stack G over time points [n, n_t, n_out, n_theta]--> [n, n_t*n_out, n_theta]
    Gaug = jnp.stack([jnp.concatenate(Gi, 0) for Gi in G])
    return block_diag(*[BetaInv] * n_t) + jnp.einsum("nki,ij,nlj->nkl", Gaug, Ainv, Gaug)


# determine what Alpha needs to be in order for A = H + diag(Alpha) to be positive definite
def make_pos_def(A, Alpha, beta=1e-8):
    # make sure matrix is not already NaN
    assert not jnp.isnan(A).any(), "Matrix contains NaN, cannot make positive definite."

    # alpha needs to be non-zero
    Alpha = np.clip(Alpha, beta, np.inf)

    # initial amount to add to matrix
    tau = 0.

    # use cholesky decomposition to check positive-definiteness of A
    while jnp.isnan(jnp.linalg.cholesky(A + tau * jnp.diag(Alpha))).any():
        # increase prior precision
        print("adding regularization")
        tau = max([2. * tau, beta])

    # tau*jnp.diag(Alpha) is the amount that needed to be added for A to be P.D.
    # make alpha = (1 + tau)*alpha so that alpha + H is P.D.
    return A + tau * jnp.diag(Alpha), (1. + tau) * Alpha
