from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from objective import Objective


def comp_contrasts(n):
    """
    """
    # (y_2 - y_1, y_3 - y_1, ..., y_n - y_1)
    C = np.concatenate([np.ones([n-1, 1]),
                        -1 * np.eye(n-1)],
                        axis=1)

    # orthogonalize contrats
    U, Sigma, Vt = np.linalg.svd(C)
    V = Vt[:-1, :]

    return(V)


def comp_fit_cov(feems, lamb_l2, lamb_smth, lamb_log, projection=True, include_var=False, ind_level=False):
    # create obj
    obj = Objective(feems.sumstats, feems.graph)
    obj.lamb_l2 = lamb_l2
    obj.lamb_smth = lamb_smth
    obj.lamb_log = lamb_log
    
    # update laplacian
    obj.graph.comp_lap(obj.graph.w)
    
    # update nll and grad
    obj.inv()
    obj.grad()
    
    if ind_level is True:
        # number of individuals
        n, p = obj.sumstats.data.shape
        
        # row index of assn matrix
        row = np.arange(n)

        # col index of assn matrix
        col = obj.graph.obs_ids        
        
        # fill up J
        J = sp.csc_matrix((np.ones(n), (row, col)), shape=(n, obj.graph.d))[:, obj.graph.perm_ids][:,:obj.graph.o]
        
        # diagonal component
        q_full_samples = 1.0 / obj.sumstats.s2 * np.ones(obj.graph.o)

        # fitted covariance
        fit_cov = J @ (obj.Linv_block['oo'] - 1/obj.graph.d) @ J.T + np.diag(J @ (1./q_full_samples))

        # empirical covariance
        emp_cov = obj.sumstats.data @ obj.sumstats.data.T / p        
    else:
        # fitted covariance
        fit_cov = obj.Linv_block['oo'] - 1/obj.graph.d + obj.sumstats.q_inv_diag.toarray()
    
        # empirical covariance
        emp_cov = obj.sumstats.S
    
    # project to the space of contrast
    if projection is True:
        C = comp_contrasts(n) if ind_level is True else comp_contrasts(obj.graph.o)
        fit_cov = C @ fit_cov @ C.T
        emp_cov = C @ emp_cov @ C.T

    if include_var is True:
        return(np.triu(fit_cov, k=0), np.triu(emp_cov, k=0))
    else:
        return(np.triu(fit_cov, k=1), np.triu(emp_cov, k=1))
