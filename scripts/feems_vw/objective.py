from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp


class Objective(object):

    def __init__(self, sumstats, graph):
        """Gradients and evaluations of the objective function
        """
        self.sumstats = sumstats
        self.graph = graph

    def _rank_one_solver(self, B):
        """Solver for linear system (L_{d-o,d-o} + ones/d) * X = B
        using rank ones update equation
        """
        # vector of ones with size d-o
        ones = np.ones(self.graph.d - self.graph.o)

        # sparse cholesky factorization
        # solve the systems
        # L_block{dd}\B
        ### TODO: how to handle when B is sparse ###
        U = self.graph.factor(B)

        # L_block{dd}\ones
        v = self.graph.factor(ones)

        # denominator
        denom = self.graph.d + np.sum(v)
        X = U - np.outer(v, v @ B) / denom

        return((X, v, denom))

    def _solve_lap_sys(self):
        """Solve (L_{d-o,d-o} + ones/d) * X = L_{d-o,o} + ones/d using rank one solver
        """
        # set B = L_{d-o,o}
        B = self.graph.L_block['do']

        # solve (L_{d-o,d-o} + ones/d) \ B
        self.lap_sol, v, denom = self._rank_one_solver(B.toarray())

        # compute rank one update for vector of ones
        ones = np.ones(self.graph.o)
        self.lap_sol += np.outer(v, ones) * (1. / self.graph.d - np.sum(v) / (self.graph.d * denom))

    def _comp_mat_block_inv(self):
        """Computes matrix block inversion formula
        """
        # Equation (4) in the note
        # multiply L_{o,d-o} by solution of lap-system
        A = self.graph.L_block['od'] @ self.lap_sol

        # multiply one matrix by solution of lap-system
        B = np.outer(np.ones(self.graph.o), self.lap_sol.sum(axis=0)) / self.graph.d

        # sum up with L_{o,o} and one matrix
        self.L_double_inv = self.graph.L_block['oo'].toarray() + 1. / self.graph.d - A - B

    def _comp_inv_lap(self, B=None):
        """Computes submatrices of inverse of lap
        """
        if B is None:
            B = np.eye(self.graph.o)

        # inverse of graph laplacian
        # compute o-by-o submatrix of inverse of lap
        self.Linv_block = {}
        self.Linv_block['oo'] = np.linalg.solve(self.L_double_inv, B)

        # compute (d-o)-by-o submatrix of inverse of lap
        # Equation (5) in the note
        self.Linv_block['do'] = -self.lap_sol @ self.Linv_block['oo']

        # stack the submatrices
        self.Linv = np.vstack((self.Linv_block['oo'], self.Linv_block['do']))

    def _comp_inv_cov(self, B=None):
        """Computes inverse of the covariance matrix
        """
        # helper
        # Equation (3) in the note
        A = -self.sumstats.q_inv_diag.toarray() - (self.sumstats.q_inv_diag @ self.L_double_inv) @ self.sumstats.q_inv_diag
        if B is None:
            B = np.eye(self.graph.o)

        # solve o-by-o linear system to get X
        self.X = np.linalg.solve(A, B)

        # inverse covariance matrix
        self.inv_cov = self.X + np.diag(self.sumstats.q)
        self.inv_cov_sum = self.inv_cov.sum(axis=0)
        self.denom = self.inv_cov_sum.sum()

    def _comp_grad_obj(self):
        """Computes the gradient of the objective function with respect to the latent variables
        dLoss / dL
        """
        # compute inverses
        self._comp_inv_lap()

        self.comp_B = self.inv_cov - (1/self.denom) * np.outer(self.inv_cov_sum, self.inv_cov_sum)
        self.comp_A = self.comp_B @ self.sumstats.S @ self.comp_B
        M = self.comp_A - self.comp_B
        self.grad_obj_L = self.sumstats.p * (self.Linv @ M @ self.Linv.T)

    def _comp_grad_reg(self):
        """Computes gradient
        """
        # gradient of nll
        gradD = np.diag(self.grad_obj_L) @ self.graph.S
        gradW = 2*self.grad_obj_L[self.graph.nnz_idx_perm] # use symmetry
        self.grad_obj = gradD - gradW
        self.grad_obj += self.lamb_l2 * self.graph.w

        # gradient of pens
        self.grad_pen = self.lamb_smth * self.graph.Delta.T @ self.graph.Delta @ (self.graph.w + self.lamb_log * np.log(self.graph.w))

    def inv(self):
        """Computes relevant inverses for gradient computations
        """
        # compute inverses
        self._solve_lap_sys()
        self._comp_mat_block_inv()
        self._comp_inv_cov()

    def grad(self, reg=True):
        """Computes relevent gradients the objective
        """
        # compute derivatives
        self._comp_grad_obj()
        if reg is True:
            self._comp_grad_reg()

    def neg_log_lik(self):
        """Evaluate the negative log-likelihood function given the current params
        """
        self.trA = self.sumstats.S @ self.inv_cov

        # trace
        self.trB = self.inv_cov_sum @ self.trA.sum(axis=1)
        self.tr = np.trace(self.trA) - self.trB / self.denom

        # det
        E = self.X + np.diag(self.sumstats.q)
        self.det = np.linalg.det(E) * self.graph.o / self.denom

        # negative log-likelihood
        nll = self.sumstats.p * (self.tr - np.log(self.det))

        return(nll)

    def loss(self):
        """Evaluate the loss function given the current params
        """
        lik = self.neg_log_lik()
        pen = .5 * self.lamb_l2 * np.linalg.norm(self.graph.w, 2)**2
        pen += .5 * self.lamb_smth * np.linalg.norm(self.graph.Delta @ (self.graph.w + self.lamb_log * np.log(self.graph.w)), 2)**2
        loss = lik + pen

        return(loss)


def neg_log_lik_w0_s2(z, obj):
     """Computes negative log likelihood for a constant w and residual variance
     """
     theta = np.exp(z)
     obj.lamb_l2 = 0.0
     obj.lamb_smth = 0.0
     obj.lamb_log = 0.0
     obj.graph.w = theta[0] * np.ones(obj.graph.edge.shape[0])
     obj.sumstats.q = obj.graph.n_samples_per_node / theta[1]
     obj.sumstats.q_diag = sp.diags(obj.sumstats.q).tocsc()
     obj.sumstats.q_inv_diag = sp.diags(1. / obj.sumstats.q).tocsc()
     obj.graph.comp_lap(obj.graph.w)
     obj.inv()
     nll = obj.neg_log_lik()

     return(nll)


def loss_wrapper(z, obj):
    """Wrapper function to optimize z=log(w) which returns the loss and gradient
    """
    obj.graph.comp_lap(np.exp(z))
    obj.inv()
    obj.grad()

    # loss / grad
    l = obj.loss()
    g = obj.grad_obj * obj.graph.w + obj.grad_pen * (obj.graph.w + obj.lamb_log)

    return((l, g))

def neg_log_lik_wrapper(z, obj):
    """
    """
    obj.graph.comp_lap(np.exp(z))
    obj.inv()
    nll = obj.neg_log_lik()

    return(nll)
