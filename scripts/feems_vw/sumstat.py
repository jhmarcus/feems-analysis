from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp


class SummaryStatistics(object):

    def __init__(self, data, graph):
        """
        """
        # genotypes
        self.data = data

        # setup train / validation sets
        self.p = data.shape[1]

        # estimate sumstats
        self._estimate_means(data,
                             graph.o,
                             graph.perm_ids,
                             graph.obs_ids,
                             graph.n_samples_per_node)

        self._estimate_precision(graph.n_samples_per_node)

        # compute sample covariance matrix
        self.S = self.Y @ self.Y.T / self.p     
        
        # compute genetic distance matrix
        n = self.S.shape[0]
        d = np.diag(self.S).reshape(-1, 1)
        ones = np.ones((n, 1))
        self.D = d @ ones.T + ones @ d.T - 2 * self.S   

    def _estimate_means(self, data, o, perm_ids, obs_ids, n_samples_per_node):
        """
        """
        # create the data matrix of means
        self.Y = np.empty((o, self.p))

        # loop of the observed nodes in order of the permuted nodes
        for i, node_id in enumerate(perm_ids[:o]):

            # find the samples assigned to the ith node
            idx = obs_ids == node_id

            # compute mean at each node
            self.Y[i, :] = np.mean(data[idx, :], axis=0)

    def _estimate_precision(self, n_samples_per_node, s2=1.0):
        """
        """
        # self.s2 = np.var(self.Y)
        # self.s2 = 1.0/2.9
        self.s2 = s2
        self.q = n_samples_per_node / self.s2
        self.q_diag = sp.diags(self.q).tocsc()
        self.q_inv_diag = sp.diags(1./self.q).tocsc()
