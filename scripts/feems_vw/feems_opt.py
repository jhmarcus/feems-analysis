from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar, minimize, fmin_l_bfgs_b

from sumstat import SummaryStatistics
from graph import Graph
from objective import Objective, loss_wrapper, neg_log_lik_wrapper, neg_log_lik_w0_s2
from utils import soft_thresh


class FEEMS(object):

    def __init__(self, data, coord, grid, edge):
        """Fast Estimation of Effective Migration Surfaces is a method for
        vizualizing non-stationary spatial structure on a geographic map
        Arguments

        """
        # construct the graph
        self.graph = Graph(coord=coord, grid=grid, edge=edge)

        # compute summary statistics
        self.sumstats = SummaryStatistics(data=data, graph=self.graph)

    def fit_w0_s2(self, verbose=True):
        """Fits a constant migration surface (no reg) and residual variance by maximum likelihood
        """
        obj = Objective(self.sumstats, self.graph)
        res = minimize(neg_log_lik_w0_s2,
                    [0.0, 0.0],
                    method="Nelder-Mead",
                    args=(obj))
        assert res.success is True, "did not converge"
        w0_hat = np.exp(res.x[0])
        s2_hat = np.exp(res.x[1])
        self.graph.w0 = w0_hat * np.ones(self.graph.w.shape[0])
        self.sumstats.s2 = s2_hat
        self.sumstats.q = self.graph.n_samples_per_node / s2_hat
        self.sumstats.q_diag = sp.diags(self.sumstats.q).tocsc()
        self.sumstats.q_inv_diag = sp.diags(1. / self.sumstats.q).tocsc()

        # print update
        self.train_loss, _ = loss_wrapper(np.log(self.graph.w0), obj)
        if verbose:
            sys.stdout.write(("constant-w/variance fit, "
                           "converged in {} iterations, "
                           "train_loss={:.7f}\n"
                          ).format(res.nfev, self.train_loss))

    def fit_quasi_newton(self, lamb_l2, lamb_smth, lamb_log, w_init, factr=1e7,
                         maxls=50, m=10, lb=-np.Inf, ub=np.Inf, maxiter=15000, 
                         verbose=True):
        """
        """
        obj = Objective(self.sumstats, self.graph)
        obj.lamb_l2 = lamb_l2
        obj.lamb_smth = lamb_smth
        obj.lamb_log = lamb_log
        res = fmin_l_bfgs_b(func=loss_wrapper,
                            x0=np.log(w_init),
                            args=[obj],
                            factr=factr,
                            m=m,
                            maxls=maxls,
                            maxiter=maxiter,
                            approx_grad=False,
                            bounds=[(lb, ub) for _ in range(obj.graph.Delta.shape[1])])
        if maxiter >= 100:
            assert res[2]["warnflag"] == 0, "did not converge"
        self.graph.w = np.exp(res[0])

        # print update
        self.train_loss, _ = loss_wrapper(res[0], obj)
        self.train_nll = neg_log_lik_wrapper(res[0], obj)
        self.pen = self.train_loss - self.train_nll
        if verbose:
            sys.stdout.write(("lambda_l2={:.7f}, "
                              "lambda={:.7f}, "
                              "alpha={:.7f}, "
                              "converged in {} iterations, "
                              "train_loss={:.7f}\n"
                             ).format(lamb_l2,
                                      lamb_smth,
                                      lamb_log,
                                      res[2]["nit"],
                                      self.train_loss))

    def compute_fitted_distance(self):
        """Compute a fitted genetic distance with current estimate of w in the graph
        """
        # update lap
        self.graph.comp_lap(self.graph.w)

        # number of individuals and features
        n, p = self.sumstats.data.shape
        row = np.arange(n)  # row index of assn matrix
        col = self.graph.obs_ids # col index of assn matrix
        J = sp.csc_matrix((np.ones(n), (row, col)), shape=(n, self.graph.d))[:, self.graph.perm_ids]

        # fitted covariance
        q_full_samples = 1.0 / self.sumstats.s2 * np.ones_like(self.sumstats.q)
        Shat = J @ np.linalg.pinv(self.graph.L.toarray()) @ J.T + np.diag(J[:,:self.graph.o] @ (1./q_full_samples))
        d = np.diag(Shat).reshape(-1, 1)
        ones = np.ones((n, 1))
        Dhat = d @ ones.T + ones @ d.T - 2 * Shat
        return(Dhat)
        
    def _w_update(self):
        """Update the migration values
        """
        # gradient at current iterate
        w = self.graph.w
        grad = self.obj.grad_obj * w + self.obj.grad_pen * (w + self.obj.lamb_log)
        loss = self.losses[-1]

        if self.line_search is True:
            self._line_search(w, grad, loss)

        if self.line_search is False:
            self._fixed_stepsize(w, grad)

        # optimality criterion
        eta = self.eta_line_search if self.line_search else self.eta
        self.opt_crit = np.linalg.norm(np.log(self.graph.w) - np.log(w), 2) / (len(self.graph.w) * eta)

        # check stopping criterion
        if self.opt_crit <= self.eps:
            sys.stdout.write("Convergence criterion reached: Gradient norm: {:.6f} Threshold: {:.6f}\n\n".format(self.opt_crit, self.eps))
            self.converged = True

        # initialize step size
        if abs(loss - self.loss) <= 1e-50 and self.converged is False:
            sys.stdout.write("The loss value did not change: Gradient norm: {:.6f} Threshold: {:.6f}\n\n".format(self.opt_crit, self.eps))
            self.converged = True

    def _one_step_grad_desc(self, w, grad, eta):
        """Perform one step of projected gradient descent
        """
        w_next = w * np.exp(-eta * grad)
        return(self._projection_onto_ranges(w_next))

    def _projection_onto_ranges(self, w):
        """Projection onto lower- and upper bounds of migration rates
        """
        return(np.exp(np.clip(np.log(w), np.log(self.lb), np.log(self.ub))))

    def _fixed_stepsize(self, w, grad):
        """Perform fixed step size
        """
        # update
        self.graph.w = self._one_step_grad_desc(w, grad, self.eta)
        self.graph.comp_lap(self.graph.w)
        self.obj.inv()
        self.obj.grad()
        self.loss = self.obj.loss()

    # def _line_search(self, m, grad, loss):
    #     """Perform line search to find step size
    #     """
    #     self.eta_line_search = self.eta
    #     converged_line_search = False
    #     while not converged_line_search:
    #
    #         # update
    #         self.graph.m = self._one_step_grad_desc(m, grad, self.eta_line_search)
    #         self.graph.comp_lap(self.graph.m)
    #         self.obj.inv()
    #         self.loss = self.obj.loss()
    #
    #         # stopping criterion for line search
    #         criterion = loss + self.c * np.dot(np.log(self.graph.m) - np.log(m), grad * m) - self.loss
    #         if criterion < 0:
    #             self.eta_line_search *= self.rho
    #         else:
    #             converged_line_search = True
    #
    #             # update objective
    #             self.obj.grad()
    #             if self.eta_line_search <= 1e-12:
    #                 sys.stdout.write("The algorithm stops since step size becomes too small: \
    #                                     Gradient norm: {:.6f} Threshold: {:.6f}\n\n".format(self.opt_crit, self.eps))
    #                 self.converged = True

    def fit(self, n_iter, lamb_l2, lamb_smth, lamb_log, eta,
            w_init, line_search=False, lb=1e-10, ub=1e10,
            c=.2, rho=.5, eps=5*1.0e-3, n_print=1000):
        """Fits the feems model for a single regularzation parameters
        """
        # m init
        self.graph.comp_lap(w_init)

        # opt params
        self.eta = eta
        self.line_search = line_search
        self.lb = lb
        self.ub = ub
        self.c = c
        self.rho = rho
        self.eps = eps

        # compute inital objective
        self.obj = Objective(self.sumstats, self.graph)
        self.obj.lamb_l2 = lamb_l2
        self.obj.lamb_smth = lamb_smth
        self.obj.lamb_log = lamb_log

        # compute inverse covariances and gradient of objective
        self.obj.inv()
        self.obj.grad()

        ########## stopping criterion ##########
        # if self.opt_crit < self.eps terminate algorithm
        self.opt_crit = np.Inf
        self.converged = False

        # iterate
        self.losses = []
        self.losses.append(self.obj.loss())
        for i in range(n_iter):

            # variable updates
            self._w_update()
            if self.converged:
                break

            self.losses.append(self.loss)

            if i % n_print == 0:
                sys.stdout.write("iteration {}: loss = {}\n".format(i, self.losses[-1]))

            # update the graph in the objective
            self.obj.graph = self.graph

    # def warmstart(self, n_iter, lamb_grid, eta, m_init,
    #               line_search=False, lb=1e-10, ub=1e10,
    #               c=.2, rho=.5, eps=5*1.0e-3, n_print=1000):
    #     """Fits the feems model for a grid regularzation parameters with a
    #     intializing each fit using warmstart
    #     """
    #     # setup matrix of fitted values per lambda
    #     M = np.empty((lamb_grid.shape[0]+1, self.graph.d))
    #     M[0, :] = m_init
    #
    #     # warm start
    #     for i, lamb in enumerate(lamb_grid):
    #
    #         sys.stdout.write("# Running feems with lamb={:.6f}\n".format(lamb))
    #         self.fit(n_iter=n_iter, lamb=lamb, eta=eta, m_init=M[i, :],
    #                  line_search=line_search, lb=lb, ub=ub, c=c, rho=rho,
    #                  eps=eps, n_print=n_print)
    #
    #         M[i+1, :] = self.graph.m
    #
    #     return(M)

    def admm(self, n_iter, lamb_l2, lamb_smth, lamb_log, eta, w_init,
            rho=100.0, lb=1e-10, ub=1e10, eps=5*1.0e-3, n_print=1000):
        """Fits the feems model for a single regularzation parameters using admm
        """
        # m init
        self.graph.comp_lap(w_init)

        # opt params
        self.eta = eta
        self.rho = rho
        self.lb = lb
        self.ub = ub
        self.eps = eps

        # compute inital objective
        self.obj = Objective(self.sumstats, self.graph)
        self.obj.lamb_l2 = lamb_l2
        self.obj.lamb_smth = lamb_smth
        self.obj.lamb_log = lamb_log

        # compute inverse covariances and gradient of objective
        self.obj.inv()
        self.obj.grad()

        # admm variables
        z = np.zeros(self.obj.graph.Delta.shape[0])
        u = np.zeros(self.obj.graph.Delta.shape[0])

        ########## stopping criterion ##########
        # if self.opt_crit < self.eps terminate algorithm
        self.opt_crit = np.Inf
        self.converged = False

        # iterate
        self.losses = []
        self.losses.append(self.obj.loss())
        for i in range(n_iter):

            # variable updates
            # w update
            grad = self.obj.grad_obj * self.graph.w + self.rho * self.obj.graph.Delta.T @ \
                    (self.obj.graph.Delta @ (self.graph.w + self.obj.lamb_log * np.log(self.graph.w)) - z - u/rho) * \
                                                (self.obj.lamb_log + self.graph.w)
            self.graph.w = self._projection_onto_ranges(self.graph.w * np.exp(-eta * grad))
            # if self.converged:
                # break

            # z update
            z = soft_thresh(self.obj.graph.Delta @ (self.graph.w + self.obj.lamb_log * np.log(self.graph.w)) \
                                                                                    - u/self.rho, self.obj.lamb_smth/self.rho)

            # u update
            u = u + self.rho * (z - self.obj.graph.Delta @ (self.graph.w + self.obj.lamb_log * np.log(self.graph.w)))

            # update graph
            self.graph.comp_lap(self.graph.w)
            # update objective
            self.obj.inv()
            self.obj.grad()

            loss = self.obj.neg_log_lik() + self.obj.lamb_l2/2 * np.linalg.norm(self.graph.w, 2)**2 \
                    + self.obj.lamb_smth * np.linalg.norm(self.obj.graph.Delta @ (self.graph.w + self.obj.lamb_log * np.log(self.graph.w)), 1)
            self.losses.append(loss)

            if i % n_print == 0:
                sys.stdout.write("iteration {}: loss = {}\n".format(i, self.losses[-1]))

            # update the graph in the objective
            self.obj.graph = self.graph
