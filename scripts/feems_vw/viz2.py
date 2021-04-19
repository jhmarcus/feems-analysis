from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import ticker
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj, transform


class Viz(object):
    
    def __init__(self, ax, feems, weights=None,
                 projection=None, 
                 coastline_m="50m",
                 coastline_linewidth=.5,
                 sample_pt_size=1,
                 sample_pt_linewidth=.5,
                 sample_pt_color="#d9d9d9",
                 sample_pt_jitter_std=0.0,
                 sample_pt_alpha=1.0,
                 sample_pt_zorder=2,
                 obs_node_size=10,
                 obs_node_textsize=7,
                 obs_node_linewidth=.5,
                 obs_node_color="#d9d9d9",
                 obs_node_alpha=1.0,
                 obs_node_zorder=2,
                 edge_color="#d9d9d9",
                 edge_width=1,
                 edge_alpha=1.0, 
                 edge_zorder=2,
                 abs_max=2,
                 cbar_font_size=12,
                 cbar_nticks=3,
                 cbar_orientation="horizontal",
                 cbar_ticklabelsize=12,
                 cbar_width="20%",
                 cbar_height="5%",
                 cbar_loc="lower left",
                 cbar_bbox_to_anchor=(0.05, 0.2, 1, 1),
                 ell_scaler=np.sqrt(3.0) / 6.0,
                 ell_edgecolor="gray",
                 ell_lw=.2,
                 ell_abs_max=.5,
                 target_dist_pt_size=10,
                 target_dist_pt_linewidth=.5,
                 target_dist_pt_alpha=1.0,
                 target_dist_pt_zorder=2,
                 seed=1996):
        """A visualizaiton module for feems
        """
        # main attributes
        self.ax = ax
        self.ax.axis("off")
        self.feems = feems
        self.grid = feems.graph.grid
        self.coord = feems.graph.coord
        self.projection = projection
        self.seed = seed
        
        # set the random seed
        np.random.seed = self.seed 
        
        ############ viz components ############
        self.coastline_m = coastline_m
        self.coastline_linewidth = coastline_linewidth
        
        # sample pt
        self.sample_pt_size = sample_pt_size
        self.sample_pt_linewidth = sample_pt_linewidth
        self.sample_pt_color = sample_pt_color
        self.sample_pt_zorder = sample_pt_zorder
        self.samplte_pt_alpha = sample_pt_alpha
        self.sample_pt_jitter_std = sample_pt_jitter_std
        
        # obs nodes
        self.obs_node_size = obs_node_size
        self.obs_node_textsize = obs_node_textsize
        self.obs_node_alpha = obs_node_alpha
        self.obs_node_linewidth = obs_node_linewidth
        self.obs_node_color = obs_node_color
        self.obs_node_zorder = obs_node_zorder
        
        # edge
        self.edge_width = edge_width
        self.edge_alpha = edge_alpha
        self.edge_zorder = edge_zorder
        self.edge_color = edge_color
        
        # colorbar
        self.abs_max = abs_max
        self.cbar_font_size = cbar_font_size
        self.cbar_nticks = cbar_nticks
        self.cbar_orientation = cbar_orientation
        self.cbar_ticklabelsize = cbar_ticklabelsize
        self.cbar_width = cbar_width
        self.cbar_height = cbar_height
        self.cbar_loc = cbar_loc
        self.cbar_bbox_to_anchor = cbar_bbox_to_anchor
        
        
        # ellipse
        self.ell_scaler = ell_scaler
        self.ell_edgecolor = ell_edgecolor
        self.ell_abs_max = ell_abs_max
        self.ell_lw = ell_lw
        
        # target correlations
        self.target_dist_pt_size = target_dist_pt_size
        self.target_dist_pt_linewidth = target_dist_pt_linewidth
        self.target_dist_pt_alpha = target_dist_pt_alpha
        self.target_dist_pt_zorder = target_dist_pt_zorder
        
        # colors
        self.eems_colors = ["#994000", "#CC5800", "#FF8F33",
                            "#FFAD66", "#FFCA99", "#FFE6CC",
                            "#FBFBFB", "#CCFDFF", "#99F8FF",
                            "#66F0FF", "#33E4FF", "#00AACC",
                            "#007A99"]
        self.edge_cmap = clr.LinearSegmentedColormap.from_list("eems_colors",
                                                               self.eems_colors,
                                                               N=256)
        self.edge_norm = clr.LogNorm(vmin=10**(-self.abs_max), vmax=10**(self.abs_max))
        self.dist_cmap = plt.get_cmap("viridis_r")
        
        # extract node positions on the lattice
        self.idx = nx.adjacency_matrix(self.feems.graph.graph).nonzero()
        
        # edge weights
        if weights is None:
            self.weights = recover_nnz_entries(self.feems.graph)
        else:
            self.weights = weights
        self.norm_log_weights = np.log10(self.weights) - np.mean(np.log10(self.weights))
        self.n_params = int(len(self.weights) / 2)
        
        # plotting maps
        if self.projection != None:
            self.proj = Proj(projection.proj4_init)            
            self.coord = project_coords(self.coord, self.proj)
            self.grid = project_coords(self.grid, self.proj)

    def draw_map(self):
        """Draws the underlying map projection
        """
        self.ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", zorder=0)
        self.ax.coastlines(self.coastline_m, color="#636363", linewidth=self.coastline_linewidth, zorder=0)
        
    def draw_samples(self):
        """Draw the individual sample coordinates
        """ 
        jit_coord = self.coord + np.random.normal(loc=0.0, scale=self.sample_pt_jitter_std, size=self.coord.shape) 
        self.ax.scatter(jit_coord[:, 0], 
                        jit_coord[:, 1],
                        edgecolors="black",
                        linewidth=self.sample_pt_linewidth,
                        s=self.sample_pt_size,
                        alpha=self.sample_pt_alpha,
                        color=self.sample_pt_color,
                        marker=".",
                        zorder=self.sample_pt_zorder)
        
    def draw_obs_nodes(self, use_ids=False):
        """Draw the observed node coordinates
        """ 
        obs_perm_ids = self.feems.graph.perm_ids[:self.feems.graph.o]
        obs_grid = self.grid[obs_perm_ids, :]
        if use_ids:
            for i,perm_id in enumerate(obs_perm_ids):
                self.ax.text(obs_grid[i, 0], 
                             obs_grid[i, 1], 
                             str(perm_id),
                             horizontalalignment="center",
                             verticalalignment="center",
                             size=self.obs_node_textsize,
                             zorder=self.obs_node_zorder)
        else:
            self.ax.scatter(obs_grid[:, 0],
                            obs_grid[:, 1],
                            edgecolors="black",
                            linewidth=self.obs_node_linewidth,
                            s=self.obs_node_size * np.sqrt(self.feems.sumstats.q),
                            alpha=self.obs_node_alpha,
                            color=self.obs_node_color,
                            zorder=self.obs_node_zorder)
        
    def draw_edges(self, use_weights=False):
        """Draw the edges of the graph
        """
        if use_weights:
            nx.draw(self.feems.graph.graph,
                    ax=self.ax,
                    node_size=0.0,
                    edge_cmap=self.edge_cmap,
                    edge_norm=self.edge_norm,
                    alpha=self.edge_alpha,
                    pos=self.grid,
                    width=self.edge_width,
                    edgelist=list(np.column_stack(self.idx)),
                    edge_color=self.norm_log_weights,
                    edge_vmin=-self.abs_max,
                    edge_vmax=self.abs_max)  
        else:
            nx.draw(self.feems.graph.graph,
                    ax=self.ax,
                    node_size=0.0,
                    alpha=self.edge_alpha,
                    pos=self.grid,
                    width=self.edge_width,
                    edgelist=list(np.column_stack(self.idx)),
                    edge_color=self.edge_color, 
                    zorder=self.edge_zorder)
            
    def draw_edge_colorbar(self):
        """Draws colorbar
        """
        self.edge_sm = plt.cm.ScalarMappable(cmap=self.edge_cmap, norm=self.edge_norm)
        self.edge_sm._A = []                
        self.edge_tick_locator = ticker.LogLocator(base=10, numticks=self.cbar_nticks)
        self.edge_axins = inset_axes(self.ax,
                                     width=self.cbar_width,
                                     height=self.cbar_height, 
                                     loc=self.cbar_loc,
                                     bbox_to_anchor=self.cbar_bbox_to_anchor,
                                     bbox_transform=self.ax.transAxes,
                                     borderpad=0)
        self.edge_cbar = plt.colorbar(self.edge_sm, cax=self.edge_axins, orientation=self.cbar_orientation)   
        self.edge_cbar.locator = self.edge_tick_locator
        self.edge_cbar.update_ticks()
        self.edge_cbar.ax.tick_params(which="minor", length=0)
        self.edge_cbar.ax.set_title(r"log10(w)", loc="center")
        self.edge_cbar.ax.set_title(self.edge_cbar.ax.get_title(), fontsize=self.cbar_font_size)
        self.edge_cbar.ax.tick_params(labelsize=self.cbar_ticklabelsize)
            
    def draw_ellipses(self):
        """Draws ellipses for visualizing anistropy
        """    
        # create sparse W matrices
        w = self.feems.graph.w
        norm_log_w = np.log10(self.feems.graph.w) - np.mean(np.log10(self.feems.graph.w))
        W = self.feems.graph.inv_triu(w, perm=False) 
        norm_log_W = self.feems.graph.inv_triu(norm_log_w, perm=False) 
            
        # get all the cliques
        cliqs = list(nx.clique.enumerate_all_cliques(self.feems.graph.graph))
        tri = [x for x in cliqs if len(x) == 3]

        # loop over each triangle
        ells = []
        norm_log_mean_ws = []
        edgecolors = []
        for i in range(len(tri)):
    
            # node ids on each corner of triangle
            n_0 = int(tri[i][0])
            n_1 = int(tri[i][1])
            n_2 = int(tri[i][2])
                
            # extract node coords
            x_0, y_0 = self.grid[n_0, 0], self.grid[n_0, 1]
            x_1, y_1 = self.grid[n_1, 0], self.grid[n_1, 1]
            x_2, y_2 = self.grid[n_2, 0], self.grid[n_2, 1]  
            xy = (np.mean([x_0, x_1, x_2]), np.mean([y_0, y_1, y_2]))

            # extract edge weights for each triangle
            w_01 = W[n_0, n_1]
            w_12 = W[n_1, n_2]
            w_20 = W[n_2, n_0]
    
            # extract from relative weights
            norm_log_w_01 = norm_log_W[n_0, n_1]
            norm_log_w_12 = norm_log_W[n_1, n_2]
            norm_log_w_20 = norm_log_W[n_2, n_0]
            norm_log_mean_w = np.mean([norm_log_w_01, norm_log_w_12, norm_log_w_20])
            norm_log_mean_ws.append(norm_log_mean_w)
            if np.abs(norm_log_mean_w) <= self.ell_abs_max:
                edgecolors.append(self.ell_edgecolor)
            else:
                edgecolors.append("None")
                
            # setup diffusion matrix [code adapted from jnovembre's R script]
            M = np.empty((2, 3))
            delta_01 = np.array([x_1 - x_0, y_1 - y_0])
            delta_12 = np.array([x_2 - x_1, y_2 - y_1])
            delta_20 = np.array([x_2 - x_0, y_2 - y_0])
            
            # TODO: double check sqrt(w_ij) vs w_ij^2???
            M[:,0] = w_01 * delta_01
            M[:,1] = w_12 * delta_12
            M[:,2] = w_20 * delta_20
            D = M @ M.T
            lamb, U = np.linalg.eigh(D)
            lamb = lamb[::-1]
            #U = U[:, ::-1] # TODO: double check reverse sorting of eigenvectors
                
            # create ellipse objects
            l2_norm = np.linalg.norm(delta_01, 2)
            scale = lamb / lamb[0]
            conv_fact = 360.0 / (2 * math.pi)
            angle = conv_fact * np.arctan(U[1, 0] / U[0, 0]) 
            width = scale[0] * l2_norm * self.ell_scaler
            height = scale[1] * l2_norm * self.ell_scaler
            ells.append(Ellipse(xy=xy,
                                width=height, 
                                height=width,
                                angle=angle))
                
        # add the ellipse
        p = PatchCollection(ells, cmap=self.edge_cmap, lw=self.ell_lw)
        p.set_array(np.array(norm_log_mean_ws))
        p.set_edgecolor(edgecolors)
        self.ax.add_collection(p)
            
    def draw_target_dist(self, target_node_idx):
        """Draws a scatter plot of the correlation of an 
        observed target node with all other observed nodes
        """
        obs_perm_ids = self.feems.graph.perm_ids[:self.feems.graph.o]
        obs_grid = self.grid[obs_perm_ids, :]
        i = np.where(obs_perm_ids == target_node_idx)[0][0] 
        mask = np.ones(self.feems.graph.o, dtype=bool)
        mask[i] = False
        d = self.feems.sumstats.D[i, :]
                
        # make the viz
        # plot X for target pt
        self.ax.scatter(obs_grid[~mask, 0],
                        obs_grid[~mask, 1],
                        marker="x",
                        s=self.target_dist_pt_size,
                        color="black",
                        zorder=self.target_dist_pt_zorder)
        
        #  plot heatmap for all other pts
        self.dist_plot = self.ax.scatter(obs_grid[mask, 0],
                                         obs_grid[mask, 1],
                                         edgecolors="black",
                                         linewidth=self.target_dist_pt_linewidth,
                                         cmap=self.dist_cmap,
                                         s=self.target_dist_pt_size,
                                         alpha=self.target_dist_pt_alpha,
                                         c=d[mask],
                                         zorder=self.target_dist_pt_zorder)
        
    def draw_dist_colorbar(self):
        """Draws corr colorbar
        """        
        # color scale params
        self.dist_sm = plt.cm.ScalarMappable(cmap=self.dist_cmap)
        self.dist_sm._A = []
        
        # colorbar
        self.dist_cbar = plt.colorbar(self.dist_plot,
                                      fraction=self.cbar_frac,
                                      pad=self.cbar_pad,
                                      orientation=self.cbar_orientation)
        self.dist_cbar.ax.set_title("observed genetic distance", loc="center")
        self.dist_cbar.ax.set_title(self.dist_cbar.ax.get_title(), fontsize=self.cbar_font_size)
        self.dist_tick_locator = ticker.LinearLocator(numticks=self.cbar_nticks)
        self.dist_cbar.locator = self.dist_tick_locator
        self.dist_cbar.update_ticks()
        
            
def recover_nnz_entries(graph):
    """Permute W matrix and vectorize according to the CSC index format
    """
    W = graph.inv_triu(graph.w, perm=False)
    w = np.array([])
    idx = nx.adjacency_matrix(graph.graph).nonzero()
    idx = list(np.column_stack(idx))
    for i in range(len(idx)):
        w = np.append(w, W[idx[i][0],idx[i][1]])
    return(w)


def project_coords(X, proj):
    """Project coordinates
    """
    P = np.empty(X.shape)
    for i in range(X.shape[0]):
        x, y =  proj(X[i, 0], X[i, 1])
        P[i, 0] = x
        P[i, 1] = y
    return(P)