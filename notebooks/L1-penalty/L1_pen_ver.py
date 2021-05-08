from __future__ import absolute_import, division, print_function

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize

from feems import Objective
sys.path.append('../node-vs-edge-params')
from node_param_ver import NodeParam_SpatialGraph


class L1Pen_SpatialGraph(NodeParam_SpatialGraph):          
    def __init__(self, genotypes, sample_pos, node_pos, edges, scale_snps=True):
        """Inherit from the object NodeParam_SpatialGraph and add methods for 
        estimation of edge weights/graph nodes with L1-norm penalty
        """          
        super().__init__(genotypes=genotypes,
                         sample_pos=sample_pos,
                         node_pos=node_pos,
                         edges=edges,
                         scale_snps=scale_snps)     
        
    # ------------------------- Optimizers -------------------------
    
    def fit_admm(
        self,
        lamb,
        w_init=None,
        s2_init=None,
        alpha=None,
        lb=-np.Inf,
        ub=np.Inf,
        eta=1/15000,
        rho=100.0,        
        maxiter=20000,
        eps=5*1e-5,
        verbose=True,
        n_print=1000
    ):    
        """Estimates the edge weights of the full model holding the residual
        variance fixed using ADMM.

        Args:
            lamb (:obj:`float`): penalty strength on weights
            w_init (:obj:`numpy.ndarray`): initial value for the edge weights
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log weights
            lb (:obj:`int`): lower bound of weights
            ub (:obj:`int`): upper bound of weights
            eta (:obj:`float`): step size for the ADMM algorithm
            rho (:obj:`float`): reg parameter for ADMM
            maxiter (:obj:`int`): maximum number of iterations to run ADMM
            eps (:obj:`float`): tolerance for convergence
            verbose (:obj:`Bool`): boolean to print summary of results
        """              
        # init from null model if no init weights are provided
        if w_init is None and s2_init is None:
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
            self.comp_precision(s2=s2_init)

        # prefix alpha if not provided
        if alpha is None:
            alpha = 1.0 / self.w0.mean()
        else:
            assert type(alpha) == float, "alpha must be float"
            assert alpha >= 0.0, "alpha must be non-negative"
            
        # compute initial objective
        obj = Objective(self)
        obj.lamb = lamb
        obj.alpha = alpha
        obj.sp_graph.comp_graph_laplacian(w_init)
        obj.inv()
        obj.grad()

        # admm dummy variables
        z = np.zeros(obj.sp_graph.Delta.shape[0])
        u = np.zeros(obj.sp_graph.Delta.shape[0])

        ########## stopping criterion ##########
        # if self.opt_crit < self.eps terminate algorithm
        opt_crit = np.Inf
        converged = False

        # iterate
        train_losses = []
        for i in range(maxiter):
            # variable updates
            # w update
            w = self.w
            grad = obj.grad_obj * w + rho * self.Delta.T @ (self.Delta @ (obj.alpha * w + np.log(1.0 - np.exp(-obj.alpha * w))) - z - u/rho) * \
                    (obj.alpha + obj.alpha * np.exp(-obj.alpha * w) / (1.0 - np.exp(-alpha * w))) * w
            w_next = self._projection_onto_ranges(w * np.exp(-eta * grad), lb, ub)
            
            # update graph
            obj.sp_graph.comp_graph_laplacian(w_next)
            
            # update objective
            obj.inv()
            obj.grad()            
            
            # convergence criterion
            opt_crit = np.linalg.norm(np.log(w_next) - np.log(w), 2) / (len(w_next) * eta)
            if opt_crit <= eps:
                sys.stdout.write("Convergence criterion reached: Gradient norm: {:.6f} Threshold: {:.6f}\n\n".format(opt_crit, eps))
                converged = True            

            if converged:
                break

            # z update
            z = self._soft_thresh(self.Delta @ (obj.alpha * w_next + np.log(1.0 - np.exp(-obj.alpha * w_next))) - u/rho, obj.lamb/rho)

            # u update
            u = u + rho * (z - self.Delta @ (obj.alpha * w_next + np.log(1.0 - np.exp(-obj.alpha * w_next))))
            
            # compute loss
            loss = obj.neg_log_lik() + \
                      obj.lamb * np.linalg.norm(self.Delta @ (obj.alpha * w_next + np.log(1.0 - np.exp(-obj.alpha * w_next))), 1)
            # save loss
            train_losses.append(loss)

            if verbose and i % n_print == 0:
                sys.stdout.write("iteration {}: loss = {}\n".format(i, train_losses[-1]))

            # update the graph in the objective
            obj.sp_graph = self
                        
    def fit_admm_m(
        self,
        lamb,
        m_init=None,
        s2_init=None,
        alpha=None,
        lb=-np.Inf,
        ub=np.Inf,
        eta=1/15000,
        rho=100,        
        maxiter=20000,
        eps=5*1e-5,
        verbose=True,
        n_print=1000
    ):    
        """Estimates the graph nodes of the full model holding the residual
        variance fixed using ADMM.

        Args:
            lamb (:obj:`float`): penalty strength on nodes
            w_init (:obj:`numpy.ndarray`): initial value for the graph nodes
            s2_init (:obj:`int`): initial value for s2
            alpha (:obj:`float`): penalty strength on log nodes
            lb (:obj:`int`): lower bound of nodes
            ub (:obj:`int`): upper bound of nodes
            eta (:obj:`float`): step size for the ADMM algorithm
            rho (:obj:`float`): reg parameter for ADMM
            maxiter (:obj:`int`): maximum number of iterations to run ADMM
            eps (:obj:`float`): tolerance for convergence
            verbose (:obj:`Bool`): boolean to print summary of results
        """              
        # init from null model if no init nodes are provided
        if m_init is None and s2_init is None:
            # fit null model to estimate the residual variance and init nodes
            self.fit_null_model_m(verbose=verbose)            
            m_init = self.m0
        else:
            # check initial graph nodes
            assert m_init.shape[0] == len(self), (
                "nodes must have shape of nodes"
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
            
        # compute initial objective
        obj = Objective(self)
        obj.lamb = lamb
        obj.alpha = alpha
        obj.sp_graph.comp_graph_laplacian(m_init)
        obj.inv()
        obj.grad()

        # admm dummy variables
        z = np.zeros(obj.sp_graph.Delta.shape[0])
        u = np.zeros(obj.sp_graph.Delta.shape[0])

        ########## stopping criterion ##########
        # if self.opt_crit < self.eps terminate algorithm
        opt_crit = np.Inf
        converged = False

        # iterate
        train_losses = []
        for i in range(maxiter):
            # variable updates
            # m update
            m = self.m
            w = self.B @ m
            grad = (self.B.T @ obj.grad_obj) * m + rho * self.B.T @ (self.Delta.T @ (self.Delta @ (obj.alpha * w + np.log(1.0 - np.exp(-obj.alpha * w))) - z - u/rho) * \
                    (obj.alpha + obj.alpha * np.exp(-obj.alpha * w) / (1.0 - np.exp(-alpha * w)))) * m
            m_next = self._projection_onto_ranges(m * np.exp(-eta * grad), lb, ub)
            w_next = self.B @ m_next
            
            # update graph
            obj.sp_graph.comp_graph_laplacian(m_next)
            
            # update objective
            obj.inv()
            obj.grad()            
            
            # convergence criterion
            opt_crit = np.linalg.norm(np.log(m_next) - np.log(m), 2) / (len(m_next) * eta)
            if opt_crit <= eps:
                sys.stdout.write("Convergence criterion reached: Gradient norm: {:.6f} Threshold: {:.6f}\n\n".format(opt_crit, eps))
                converged = True            

            if converged:
                break

            # z update
            z = self._soft_thresh(self.Delta @ (obj.alpha * w_next + np.log(1.0 - np.exp(-obj.alpha * w_next))) - u/rho, obj.lamb/rho)

            # u update
            u = u + rho * (z - self.Delta @ (obj.alpha * w_next + np.log(1.0 - np.exp(-obj.alpha * w_next))))
            
            # compute loss
            loss = obj.neg_log_lik() + \
                      obj.lamb * np.linalg.norm(self.Delta @ (obj.alpha * w_next + np.log(1.0 - np.exp(-obj.alpha * w_next))), 1)
            # save loss
            train_losses.append(loss)

            if verbose and i % n_print == 0:
                sys.stdout.write("iteration {}: loss = {}\n".format(i, train_losses[-1]))

            # update the graph in the objective
            obj.sp_graph = self            
 
    def _projection_onto_ranges(self, x, lb, ub):
        """Projection onto lower- and upper bounds of migration rates
        """
        return np.clip(x, lb, ub)
         
    def _soft_thresh(self, x, nu):
        """Soft threshold for lasso solution

        Arguments
        ---------
        x : np.array
            TODO: description

        nu :
            TODO: description

        Returns
        -------

        """
        return(np.sign(x) * np.maximum(abs(x) - nu, 0))            
            
            