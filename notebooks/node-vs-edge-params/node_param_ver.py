from __future__ import absolute_import, division, print_function

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize

from feems import Objective
from feems import SpatialGraph


def neg_log_lik_m0_s2(z, obj):
    """Computes negative log likelihood for a constant m and residual variance"""
    theta = np.exp(z)
    obj.lamb = 0.0
    obj.alpha = 1.0
    obj.sp_graph.m = theta[0] * np.ones(len(obj.sp_graph))
    obj.sp_graph.comp_graph_laplacian(obj.sp_graph.m)
    obj.sp_graph.comp_precision(s2=theta[1])
    obj.inv()
    nll = obj.neg_log_lik()
    return nll   


def loss_wrapper_m(z, obj):
    """Wrapper function to optimize z=log(m) which returns the loss and gradient"""
    theta = np.exp(z)
    obj.sp_graph.comp_graph_laplacian(theta)
    obj.inv()
    obj.grad()

    # loss / grad
    loss = obj.loss()
    grad = obj.sp_graph.B.T @ (obj.grad_obj + obj.grad_pen) * obj.sp_graph.m
    return (loss, grad)


class NodeParam_SpatialGraph(SpatialGraph):   
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True):
        """Inherit from the feems object SpatialGraph and add methods for 
        estimation of graph nodes
        """             
        super().__init__(genotypes=genotypes,
                         sample_pos=sample_pos,
                         node_pos=node_pos,
                         edges=edges,
                         scale_snps=scale_snps)     
        
    # ------------------------- Optimizers -------------------------
    
    def fit_null_model_m(self, verbose=True):
        """Estimates of the graph nodes and residual variance
        under the model that all the graph nodes have the same value
        """
        obj = Objective(self)
        res = minimize(neg_log_lik_m0_s2, [0.0, 0.0], method="Nelder-Mead", args=(obj))
        assert res.success is True, "did not converge"
        m0_hat = np.exp(res.x[0])
        s2_hat = np.exp(res.x[1])
        self.m0 = m0_hat * np.ones(len(self))
        self.s2 = s2_hat
        self.comp_precision(s2=s2_hat)

        # print update
        self.train_loss = neg_log_lik_m0_s2(np.r_[np.log(m0_hat), np.log(s2_hat)], obj)
        if verbose:
            sys.stdout.write(
                (
                    "constant-w/variance fit, "
                    "converged in {} iterations, "
                    "train_loss={:.7f}\n"
                ).format(res.nfev, self.train_loss)
            )    
            
    def fit_m(
        self,
        lamb,
        m_init=None,
        s2_init=None,
        alpha=None,
        factr=1e7,
        maxls=50,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
    ):
        """Estimates the graph nodes of the full model holding the residual
        variance fixed using a quasi-newton algorithm, specifically L-BFGS.

        Args:
            lamb (:obj:`float`): penalty strength on nodes
            m_init (:obj:`numpy.ndarray`): initial value for the graph nodes
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log nodes
            factr (:obj:`float`): tolerance for convergence
            maxls (:obj:`int`): maximum number of line search steps
            m (:obj:`int`): the maximum number of variable metric corrections
            lb (:obj:`int`): lower bound of log nodes
            ub (:obj:`int`): upper bound of log nodes
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

        # init from null model if no init nodes are provided
        if m_init is None and s2_init is None:
            # fit null model to estimate the residual variance and init nodes
            self.fit_null_model_m(verbose=verbose)            
            m_init = self.m0
        else:
            # check initial graph nodes
            assert m_init.shape[0] == len(self), (
                "nodes must have size of graph"
            )
            assert np.all(m_init > 0.0), "nodes must be non-negative"
            self.m0 = m_init
            self.comp_precision(s2=s2_init)

        # prefix alpha if not provided
        if alpha is None:
            alpha = 1.0 / self.m0.mean()
        else:
            assert type(alpha) == float, "alpha must be float"
            assert alpha >= 0.0, "alpha must be non-negative"

        # run l-bfgs
        obj = Objective(self)
        obj.lamb = lamb
        obj.alpha = alpha
        x0 = np.log(m_init)
        res = fmin_l_bfgs_b(
            func=loss_wrapper_m,
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
        self.m = np.exp(res[0])

        # print update
        self.train_loss, _ = loss_wrapper_m(res[0], obj)
        if verbose:
            sys.stdout.write(
                (
                    "lambda={:.7f}, "
                    "alpha={:.7f}, "
                    "converged in {} iterations, "
                    "train_loss={:.7f}\n"
                ).format(lamb, alpha, res[2]["nit"], self.train_loss)
            )            
            
            
            
            
            
            
            