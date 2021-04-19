from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod
import networkx as nx
from itertools import product


class Graph(object):

    def __init__(self, coord, grid, edge):
        """Graph represents the discrete network which the data is defined on and
        stores relevant matricies / performs linear albegra routines needed the
        model and optimization

        Arugments
        ---------
        coord : 2D ndarray
            matrix of spatial coordinates for each sample

        grid : 2D ndarray
            matrix of spatial positions of each node on the spatial graph

        edge : 2D ndarray
        """
        # input data to construct the graph
        self.grid  = grid
        self.edge = edge

        ##### Create the graph object and relevant graph matrices #####
        # create the nx graph object
        self._create_graph()

        # track nonzero edges
        self.adj_base = sp.triu(nx.adjacency_matrix(self.graph), k=1)
        self.nnz_idx = self.adj_base.nonzero()
        self.n_edge = self.nnz_idx[0].shape[0]

        # adjacency matrix on the edges
        self.Delta = self._create_incidence_matrix()

        # vectorization operator on the edges
        self.diag_oper = self._create_vect_matrix()

        ##### assign samples to nodes and permute inputs #####
        self._create_assn_ind(coord)
        self._permute_nodes()

        ##### intialize cholesky factorization object to be filled in comp_lap #####
        # this is the factorization of L11
        self.factor = None

    def _create_graph(self):
        """Creates a nx graph object that takes the given graph nodes
        """
        # number of nodes in the graph
        self.d = self.grid.shape[0]

        # construct graph from edge list
        self.graph = nx.Graph()
        self.graph.add_nodes_from(np.arange(self.d))
        self.graph.add_edges_from((self.edge-1).tolist())

        # add spatial coordinates to node attributes
        for i in range(self.d):
            self.graph.nodes[i]["pos"] = self.grid[i, :]

    def _create_assn_ind(self, coord):
        """Creates an array that assigns samples to nodes note this only needs
        to be created once
        """
        # input data to assign samples
        self.coord  = coord

        # intialize the count attribute
        self.counts = np.zeros(self.d, "int")

        # number of indivduals
        n = self.coord.shape[0]
        self.obs_ids = np.zeros(n, "int")

        for i in range(n):
            # find the node that has the minimum distance to the observed node
            idx = np.argmin(np.sum((self.coord[i, :] - self.grid)**2, axis=1))
            self.obs_ids[i] = idx
            self.counts[idx] += 1

        # number of observed demes
        self.o = np.sum(self.counts != 0) # number of observed demes

    def _permute_nodes(self):
        """Permutes all graph matricies to start with the observed nodes first
        and then the unobserved nodes
        """
        # indicies of all nodes
        self.ids = np.arange(self.d)

        # permuted node ids
        self.perm_ids = np.concatenate([self.ids[self.counts != 0],
                                        self.ids[self.counts == 0]])

        # number of samples on the permuted nodes
        self.n_samples_per_node = self.counts[self.perm_ids][:self.o]

        # construct adj matrix with permuted nodes
        row = self.perm_ids.argsort()[self.nnz_idx[0]]
        col = self.perm_ids.argsort()[self.nnz_idx[1]]
        self.nnz_idx_perm = (row, col)
        self.adj_perm = sp.coo_matrix((np.ones(self.n_edge), (row, col)), shape=(self.d,self.d))

        # permute diag operator
        vect_idx_r = row + self.d*col
        vect_idx_c = col + self.d*row
        self.S = self.diag_oper[:,vect_idx_r] + self.diag_oper[:,vect_idx_c]

    def comp_lap(self, weight, perm=True):
        """Computes the graph laplacian note this is computed each step of the
        optimization so needs to be fast
        """
        if 'array' in str(type(weight)):
            self.w = weight
            self.W = self.inv_triu(self.w, perm=perm)
        elif 'matrix' in str(type(weight)):
            self.W = weight
        else:
            print('inaccurate argument')
        W_rowsum = np.array(self.W.sum(axis=1)).reshape(-1)
        self.D = sp.diags(W_rowsum).tocsc()
        self.L = self.D - self.W
        self.L_block = {'oo': self.L[:self.o, :self.o],
                        'dd': self.L[self.o:, self.o:],
                        'do': self.L[self.o:, :self.o],
                        'od': self.L[:self.o, self.o:]}

        if self.factor == None:
            # intialize the object if the cholesky factorization has not been
            # computed yet. This will perform the fill-in reducing permuation and
            # the cholesky factorizaton which is "slow" intially
            self.factor = cholmod.cholesky(self.L_block['dd'])
        else:
            # if it has been computed we can quickly update the facortization by
            # calling the cholesky method of factor which does not perform the
            # fill-in reducing permuation again because the sparsity pattern
            # of L11 is fixed throughout the algorithm
            self.factor = self.factor.cholesky(self.L_block['dd'])

    def inv_triu(self, w, perm=True):
        """Take upper triangluar vector as input and return symmetric weight sparse matrix
        """
        if perm==True:
            W = self.adj_perm.copy()
        else:
            W = self.adj_base.copy()
        W.data = w
        W = W + W.T
        return(W.tocsc())

    def _create_incidence_matrix(self):
        '''Create a signed incidence matrix on the edges note this is computed only once
        '''
        data = np.array([], dtype=np.float)
        row_idx = np.array([], dtype=np.int)
        col_idx = np.array([], dtype=np.int)
        n_count = 0
        for i in range(self.n_edge):
            edge1 = np.array([self.nnz_idx[0][i], self.nnz_idx[1][i]])
            for j in range(i+1, self.n_edge):
                edge2 = np.array([self.nnz_idx[0][j], self.nnz_idx[1][j]])
                if len(np.intersect1d(edge1, edge2)) > 0:
                    data = np.append(data, 1)
                    row_idx = np.append(row_idx, n_count)
                    col_idx = np.append(col_idx, i)

                    data = np.append(data, -1)
                    row_idx = np.append(row_idx, n_count)
                    col_idx = np.append(col_idx, j)

                    n_count += 1
        Delta = sp.csc_matrix((data, (row_idx, col_idx)), shape=(int(len(data)/2), self.n_edge))
        return(Delta)

    def _create_vect_matrix(self):
        '''Construct matrix operators S so that S*vec(W) is the degree vector
        note this is computed only once
        '''
        row_idx = np.repeat(np.arange(self.d), self.d)
        col_idx = np.array([], dtype=np.int)
        for ite, i in enumerate(range(self.d)):
            idx = np.arange(0, self.d**2, self.d) + ite
            col_idx = np.append(col_idx, idx)
        S = sp.csc_matrix((np.ones(self.d**2), (row_idx, col_idx)), shape=(self.d, self.d**2))
        return(S)
