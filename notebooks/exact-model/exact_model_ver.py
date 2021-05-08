import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_l_bfgs_b, minimize

from feems.objective import loss_wrapper
from feems import Objective
from feems import SpatialGraph


class ExactLik_Objective(Objective):
    def __init__(self, sp_graph):
        """Inherit from the feems object Objective and overwrite some methods for evaluations 
        and gradient of feems objective with exact model

        Args:
            sp_graph (:obj:`feems.SpatialGraph`): feems spatial graph object
        """
        super().__init__(sp_graph=sp_graph)        
        
        # indicator whether fitting null model or not
        self.is_null = False

    def _comp_inv_cov(self, B=None):
        """Computes inverse of the covariance matrix"""
        # helper
        A = (
            -self.q_inv_diag.toarray()
            - (self.q_inv_diag @ self.L_double_inv) @ self.q_inv_diag/4
        )
        if B is None:
            B = np.eye(self.sp_graph.n_observed_nodes)

        # solve o-by-o linear system to get X
        self.X = np.linalg.solve(A, B)

        # inverse covariance matrix
        self.inv_cov = self.X + np.diag(self.q)
        self.inv_cov_sum = self.inv_cov.sum(axis=0)
        self.denom = self.inv_cov_sum.sum()
        
    def _estimate_precision(self):
        """ADDED: Computes the residual precision matrix moved to objective class
        """
        self.q = self.sp_graph.n_samples_per_obs_node_permuted/(2 * (1 - np.diag(self.Linv_block['oo'] - 1/len(self.sp_graph))))
        self.q_diag = sp.diags(self.q).tocsc()
        self.q_inv_diag = sp.diags(1./self.q).tocsc()         

    def _comp_grad_obj(self):
        """Computes the gradient of the objective function with respect to the
        latent variables dLoss / dL
        """
        self.comp_B = self.inv_cov - (1.0 / self.denom) * np.outer(
            self.inv_cov_sum, self.inv_cov_sum
        )
        self.comp_A = self.comp_B @ self.sp_graph.S @ self.comp_B
        M = self.comp_A - self.comp_B
        self.grad_obj_L = 4 * self.sp_graph.n_snps * (self.Linv @ M @ self.Linv.T)

        # grads
        gradD = np.diag(self.grad_obj_L) @ self.sp_graph.P
        gradW = 2 * self.grad_obj_L[self.sp_graph.nnz_idx_perm]  # use symmetry
        
        # gradient for diag(Jq^-1)
        M_diag = np.diag(M)
        kron_D = self.Linv.T * self.Linv.T
        gr_obj_D = (-2 * self.sp_graph.n_snps * (M_diag/self.sp_graph.n_samples_per_obs_node_permuted) @ kron_D) @ self.sp_graph.P
        kron_W = self.Linv.T[:,self.sp_graph.nnz_idx_perm[0]] * self.Linv.T[:,self.sp_graph.nnz_idx_perm[1]]
        gr_obj_W = -4 * self.sp_graph.n_snps * (M_diag/self.sp_graph.n_samples_per_obs_node_permuted) @ kron_W
        
        self.grad_obj = gradD + gr_obj_D - gradW - gr_obj_W        

    def inv(self):
        """Computes relevant inverses for gradient computations"""
        # compute inverses
        self._solve_lap_sys()
        self._comp_mat_block_inv()
        self._comp_inv_lap()
        self._estimate_precision()        
        self._comp_inv_cov()

    def neg_log_lik(self):
        """Evaluate the negative log-likelihood function given the current
        params
        """
        L_dagger_diag = np.diag(self.Linv_block['oo'] - 1/len(self.sp_graph))
        
        if np.sum(L_dagger_diag > 1) >=1 and self.is_null == False:
            nll = 1e+10        
        else:
            o = self.sp_graph.n_observed_nodes
            self.trA = self.sp_graph.S @ self.inv_cov

            # trace
            self.trB = self.inv_cov_sum @ self.trA.sum(axis=1)
            self.tr = np.trace(self.trA) - self.trB / self.denom

            # det
            E = self.X + np.diag(self.q)
            self.det = np.linalg.det(E) * o / self.denom

            # negative log-likelihood
            nll = self.sp_graph.n_snps * (self.tr - np.log(self.det))
        return nll


def neg_log_lik_w0_s2(z, obj):
    """Computes negative log likelihood for a constant w and residual variance"""
    theta = np.exp(z)
    obj.lamb = 0.0
    obj.alpha = 1.0
    obj.sp_graph.w = theta[0] * np.ones(obj.sp_graph.size())
    obj.sp_graph.comp_graph_laplacian(obj.sp_graph.w)
    obj.inv()
    nll = obj.neg_log_lik()
    return nll

    
class ExactLik_SpatialGraph(SpatialGraph):  
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True):
        """Inherit from the feems object SpatialGraph and overwrite some methods for 
        estimation of edge weights under exact model
        """             
        super().__init__(genotypes=genotypes,
                         sample_pos=sample_pos,
                         node_pos=node_pos,
                         edges=edges,
                         scale_snps=scale_snps)        
                
    # ------------------------- Optimizers -------------------------

    def fit_null_model(self, verbose=True):
        """Estimates of the edge weights under the model that all the edge weights 
        have the same value
        """
        obj = ExactLik_Objective(self)
        obj.is_null = True
        res = minimize(neg_log_lik_w0_s2, [0.0], method="Nelder-Mead", args=(obj))
        assert res.success is True, "did not converge"
        w0_hat = np.exp(res.x[0])
        self.w0 = w0_hat * np.ones(self.w.shape[0])

        # print update
        self.train_loss = neg_log_lik_w0_s2(np.r_[np.log(w0_hat)], obj)
        if verbose:
            sys.stdout.write(
                (
                    "constant-w/variance fit, "
                    "converged in {} iterations, "
                    "train_loss={:.7f}\n"
                ).format(res.nfev, self.train_loss)
            )

    def fit(
        self,
        lamb,
        w_init=None,
        alpha=None,
        factr=1e7,
        maxls=50,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
    ):
        """Estimates the edge weights of the full model using a quasi-newton algorithm, specifically L-BFGS.
        
        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            alpha (:obj:`float`): penalty strength on log weights
            factr (:obj:`float`): tolerance for convergence
            maxls (:obj:`int`): maximum number of line search steps
            m (:obj:`int`): the maximum number of variable metric corrections
            lb (:obj:`int`): lower bound of log weights
            ub (:obj:`int`): upper bound of log weights
            maxiter (:obj:`int`): maximum number of iterations to run L-BFGS
            verbose (:obj:`Bool`): boolean to print summary of results        
        """
        # check inputs
        assert lamb >= 0.0, "lambda must be non-negative"
        assert type(lamb) == float, "lambda must be float"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"

        # init from null model if no init weights are provided
        if w_init is None:
            # fit null model to estimate the residual variance and init weights
            self.fit_null_model(verbose=verbose)            
            w_init = self.w0
        else:
            # check initial edge weights
            assert w_init.shape == self.w.shape, (
                "weights must have shape of edges"
            )
            assert np.all(w_init > 0.0), "weights must be non-negative"
            self.w0 = w_init

        # prefix alpha if not provided
        if alpha is None:
            alpha = 1.0 / self.w0.mean()
        else:
            assert type(alpha) == float, "alpha must be float"
            assert alpha >= 0.0, "alpha must be non-negative"

        # run l-bfgs
        obj = ExactLik_Objective(self)
        obj.lamb = lamb
        obj.alpha = alpha
        x0 = np.log(w_init)
        res = fmin_l_bfgs_b(
            func=loss_wrapper,
            x0=x0,
            args=[obj],
            factr=factr,
            m=m,
            maxls=maxls,
            maxiter=maxiter,
            approx_grad=False,
            bounds=[(lb, ub) for _ in range(x0.shape[0])],
        )
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge"
        self.w = np.exp(res[0])

        # print update
        self.train_loss, _ = loss_wrapper(res[0], obj)
        if verbose:
            sys.stdout.write(
                (
                    "lambda={:.7f}, "
                    "alpha={:.7f}, "
                    "converged in {} iterations, "
                    "train_loss={:.7f}\n"
                ).format(lamb, alpha, res[2]["nit"], self.train_loss)
            )    
    