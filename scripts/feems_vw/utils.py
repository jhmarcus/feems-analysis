from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import fiona
from shapely.geometry import Polygon, Point, shape, MultiPoint
from shapely.affinity import translate
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sumstat import SummaryStatistics
from copy import deepcopy


def recover_nnz_entries(graph):
    '''Permute W matrix and vectorize according to the CSC index format
    '''
    W = graph.inv_triu(graph.w, perm=False)
    w = np.array([])
    idx = nx.adjacency_matrix(graph.graph).nonzero()
    idx = list(np.column_stack(idx))
    for i in range(len(idx)):
        w = np.append(w, W[idx[i][0],idx[i][1]])
    return(w)


def soft_thresh(x, nu):
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

# The following helper functions are adapted from Ben Peters' code:
# https://github.com/NovembreLab/eems-around-the-world/blob/master/subsetter/intersect.py
def load_tiles(s):
    tiles = fiona.collection(s)
    return [shape(t['geometry']) for t in tiles]


def wrap_america(tile):
    tile = Point(tile)
    if np.max(tile.xy[0]) < -40 or \
            np.min(tile.xy[0]) < -40:
        tile = translate(tile, xoff=360.)

    return tile.xy[0][0], tile.xy[1][0]


def create_tile_dict(tiles, bpoly):
    pts = dict() #dict saving ids
    rev_pts = dict()
    edges = set()
    pts_in = dict() #dict saving which points are in region

    for c, poly in enumerate(tiles):
        x, y = poly.exterior.xy
        points = zip(np.round(x, 3), np.round(y, 3))
        points = [wrap_america(p) for p in points]
        for p in points:
            if p not in pts_in:
                pts_in[p] = bpoly.intersects(Point(p))  # check if point is in region
                if pts_in[p]:
                    pts[p] = len(pts)  # if so, give id
                    rev_pts[len(rev_pts)] = p

        for i in range(3):
            pi, pj = points[i], points[i + 1]
            if pts_in[pi] and pts_in[pj]:
                if pts[pi] < pts[pj]:
                    edges.add((pts[pi] + 1, pts[pj] + 1))
                else:
                    edges.add((pts[pj] + 1, pts[pi] + 1))

        #if c % 100 == 0:
        #    print(c, len(tiles))

    pts = [Point(rev_pts[p]) for p in range(len(rev_pts))]
    return pts, rev_pts, edges


def unique2d(a):
    x, y = a.T
    b = x + y * 1.0j
    idx = np.unique(b, return_index=True)[1]
    return a[idx]


def get_closest_point_to_sample(points, samples):
    usamples = unique2d(samples)
    dists = dict((tuple(s), np.argmin([Point(s).distance(Point(p))
                                       for p in points])) for s in usamples)

    res = [dists[tuple(s)] for s in samples]

    return np.array(res)


def prepare_input(coord, ggrid, translated, buffer=0, outer=None):
    """Prepares the graph input files for feems

    Adapted from Ben Peters eems-around-the-world repo

    Arguments
    ---------
    coord : ndarray
        n x 2 matrix of sample coordinates

    ggrid : str
        path to global grid shape file

    transform : bool
        to translate x coordinates

    buffer : float
        buffer on the convex hull of sample pts

    outer : ndarray
        q x 2 matrix of coordinates of outer polygon

    Returns
    -------
    res : tuple
        tuple of outer, edges, grid
    """
    # no outer so construct with buffer
    if outer is None:
        points = MultiPoint([(x, y) for x,y in coord])
        xy = points.convex_hull.buffer(buffer).exterior.xy
        outer = np.array([xy[0].tolist(), xy[1].tolist()]).T

    if translated:
        outer[:,0] = outer[:,0] + 360.0

    # intersect outer with discrete global grid
    bpoly = Polygon(outer)
    bpoly2 = translate(bpoly, xoff=-360.0)
    tiles2 = load_tiles(ggrid)
    tiles3 = [t for t in tiles2 if bpoly.intersects(t) or bpoly2.intersects(t)]
    pts, rev_pts, e = create_tile_dict(tiles3, bpoly)

    # construct grid array
    grid = []
    for i, v in rev_pts.items():
        grid.append((v[0], v[1]))
    grid = np.array(grid)

    assert grid.shape[0] != 0, "grid is empty changing translation"

    # un-translate
    if translated:
        pts = [Point(rev_pts[p][0] - 360.0, rev_pts[p][1]) for p in range(len(rev_pts))]
        grid[:,0] = grid[:,0] - 360.0
        outer[:,0] = outer[:,0] - 360.0

    # construct edge array
    edges = np.array(list(e))
    ipmap = get_closest_point_to_sample(pts, coord)
    
    res = (outer, edges, grid, ipmap)
    return(res)


def setup_kfold_cv(feems, n_splits=5, random_state=12):
    """Setup cross-valdiation indicies

    Arguments
    ---------
    feems : FEEMS class

    Returns
    -------
    is_train : ndarray
        n x k matrix storing boolean values determining
        if the sample is in tranining set for the kth fold
    """
    # number of individuals
    n = feems.graph.coord.shape[0]

    # splits data for cross-validation (holding out nodes)
    is_train = np.zeros((n, n_splits), dtype=bool)
    u_obs_ids = np.unique(feems.graph.obs_ids) # unique observed node indicies
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True) # k-fold cv object

    for k, (train_node_idx, val_node_idx) in enumerate(kf.split(u_obs_ids)):
        for i in range(n):
            if feems.graph.obs_ids[i] in u_obs_ids[train_node_idx]:
                is_train[i, k] = True

    return(is_train)


def create_feems_train_val(feems, is_train, fold=0):
    n_splits = is_train.shape[1]
    if fold >= n_splits:
        print('inaccurate argument')
    coord = feems.graph.coord
    genotypes = feems.sumstats.data
    
    # create feems objects
    feems_train = deepcopy(feems)
    feems_val = deepcopy(feems)

    # update feems train
    feems_train.graph._create_assn_ind(coord[is_train[:, fold], :])
    feems_train.graph._permute_nodes()
    feems_train.graph.factor = None
    feems_train.sumstats = SummaryStatistics(data=genotypes[is_train[:, fold], :], graph=feems_train.graph)
    
    # update feems val
    feems_val.graph._create_assn_ind(coord[~is_train[:, fold], :])
    feems_val.graph._permute_nodes()
    feems_val.graph.factor = None
    feems_val.sumstats = SummaryStatistics(data=genotypes[~is_train[:, fold], :], graph=feems_val.graph)

    return(feems_train, feems_val)