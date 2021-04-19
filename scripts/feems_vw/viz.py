from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import cartopy.crs as ccrs
from pyproj import Proj, transform
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon, Point, shape
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
import math


def plot_graph(ax,
               outer,
               graph,
               w,
               q,
               smooth=True,
               aniso=False,
               edge_width=1,
               cbar_font_size=12,
               abs_max=2,
               node_size=10,
               node_lw=1.0/2.0,
               node_color="#d9d9d9",
               leg=True,
               max_m=None,
               projection=None,
               cbar_pad=.05,
               cbar_frac=.02,
               coastline_m="50m",
               coastline_lw=.5,
               edge_alpha=1.0,
               smooth_alpha=1.0,
               smooth_nrow=50,
               smooth_ncol=50,
               holdout_ids=None):
        """Visualize the relative edge weights on the graph
        """
        # extract node positions on the lattice
        pos = nx.get_node_attributes(graph.graph, "pos")

        # non-zero nodes
        idx = nx.adjacency_matrix(graph.graph).nonzero()

        # get grid
        grid = graph.grid
        
        # edge grid
        edge_grid = np.array((grid[idx[0]] + grid[idx[1]]) / 2.)

        # edge weights
        weights = np.log10(w) - np.mean(np.log10(w))

        # define colormap used in EEMS borrowed from ...
        # https://github.com/halasadi/plotmaps/blob/master/R/utils.R
        eems_colors = ["#994000", "#CC5800", "#FF8F33",
                       "#FFAD66", "#FFCA99", "#FFE6CC",
                       "#FBFBFB", "#CCFDFF", "#99F8FF",
                       "#66F0FF", "#33E4FF", "#00AACC",
                       "#007A99"]

        # setup color map
        cmap = clr.LinearSegmentedColormap.from_list("eems_colors",
                                                     eems_colors,
                                                     N=256)

        # setup projection
        if projection != None:

            # redefine axes
            ax.add_feature(cfeature.LAND, facecolor="#f7f7f7")
            ax.coastlines(coastline_m, color="#636363", linewidth=coastline_lw)
            proj = Proj(projection.proj4_init)

            # project the graph coordinates
            for i in pos.keys():
                pos[i] = proj(pos[i][0], pos[i][1])

            # project observed node positions
            grid = np.empty((graph.d, 2))
            for i in range(graph.d):
                x, y =  proj(graph.grid[i, 0], graph.grid[i, 1])
                grid[i, 0] = x
                grid[i, 1] = y
                
            edge_grid_p = np.empty((edge_grid.shape[0], 2))
            for i in range(edge_grid.shape[0]):
                x, y =  proj(edge_grid[i, 0], edge_grid[i, 1])
                edge_grid_p[i, 0] = x
                edge_grid_p[i, 1] = y  
                
            # project outer positions
            outer_p = np.empty((outer.shape[0], 2))
            for i in range(outer.shape[0]):
                x, y =  proj(outer[i, 0], outer[i, 1])
                outer_p[i, 0] = x
                outer_p[i, 1] = y
        
        # draw contour
        if smooth:
            x = edge_grid_p[:,0]
            y = edge_grid_p[:,1]
            xi = np.linspace(np.min(x), np.max(x), smooth_nrow)
            yi = np.linspace(np.min(y), np.max(y), smooth_ncol)
            xi, yi = np.meshgrid(xi, yi)
            z = weights
            zi = griddata((x, y), z, (xi, yi), method="cubic")
            bpoly = Polygon(outer_p)
            mask = np.zeros_like(zi, dtype=bool)
            for i in range(zi.shape[0]):
                for j in range(zi.shape[1]):
                    p = Point((xi[i, j], yi[i,j]))                
                    if not bpoly.intersects(p):
                        mask[i, j] = True                    
            zi = np.ma.array(zi, mask=mask)
            ax.contourf(xi, 
                        yi, 
                        zi, 
                        cmap=cmap, 
                        alpha=smooth_alpha, 
                        corner_mask=True,
                        antialiased=True)
        
        # plot the edges
        nx.draw(graph.graph,
                ax=ax,
                node_size=0.0,
                edge_cmap=cmap,
                alpha=edge_alpha,
                pos=pos,
                width=edge_width,
                edgelist=list(np.column_stack(idx)),
                edge_color=weights,
                edge_vmin=-abs_max,
                edge_vmax=abs_max)
        
        # plot the observed nodes
        ax.scatter(grid[graph.perm_ids[:graph.o], 0],
                   grid[graph.perm_ids[:graph.o], 1],
                   edgecolors="black",
                   linewidth=node_lw,
                   s=node_size * np.sqrt(q),
                   color=node_color,
                   zorder=2)
        
        if aniso:
            n_edges = int(len(w) / 2)
            
            # create sparse W matrices
            W = graph.inv_triu(w[:n_edges], perm=False) 
            W_logrel = graph.inv_triu(np.log10(w[:n_edges]) - np.mean(np.log10(w[:n_edges])), perm=False) 
            
            # get all the cliques
            cliqs = list(nx.clique.enumerate_all_cliques(graph.graph))
            tri = [x for x in cliqs if len(x) == 3]

            # loop over each triangle
            ells = []
            mean_w_logrels = []
            for i in range(len(tri)):

                # node ids on each corner of triangle
                n_0 = int(tri[i][0])
                n_1 = int(tri[i][1])
                n_2 = int(tri[i][2])
                
                # extract node coords
                x_0, y_0 = grid[n_0, 0], grid[n_0, 1]
                x_1, y_1 = grid[n_1, 0], grid[n_1, 1]
                x_2, y_2 = grid[n_2, 0], grid[n_2, 1]  
                xy = (np.mean([x_0, x_1, x_2]), np.mean([y_0, y_1, y_2]))

                # extract edge weights for each triangle
                w_01 = W[n_0, n_1]
                w_12 = W[n_1, n_2]
                w_20 = W[n_2, n_0]

                # extract from relative weights
                w_01_logrel = W_logrel[tri[i][0], tri[i][1]]
                w_12_logrel = W_logrel[tri[i][1], tri[i][2]]
                w_20_logrel = W_logrel[tri[i][2], tri[i][0]]
                mean_w_logrel = np.mean([w_01_logrel, w_12_logrel, w_20_logrel])
                mean_w_logrels.append(mean_w_logrel)
                
                # setup diffusion matrix [code adapted from jnovembre's R script]
                M = np.empty((2, 3))
                delta_01 = np.array([x_1 - x_0, y_1 - y_0])
                delta_12 = np.array([x_2 - x_1, y_2 - y_1])
                delta_20 = np.array([x_2 - x_0, y_2 - y_0])
                M[:,0] = w_01 * delta_01
                M[:,1] = w_12 * delta_12
                M[:,2] = w_20 * delta_20
                D = M @ M.T
                lamb, U = np.linalg.eigh(D)
                lamb = lamb[::-1]
                U = U[:, ::-1]
                
                # create ellipse objects
                l2_norm = np.linalg.norm(delta_01, 2)
                scale = lamb / lamb[0]
                angle = (360.0 / (2 * math.pi)) * np.arctan(U[1, 0] / U[0, 0]) 
                width = scale[0] * l2_norm * (np.sqrt(3.0) / 6.0)
                height = scale[1] * l2_norm* (np.sqrt(3.0) / 6.0)
                ells.append(Ellipse(xy=xy,
                                    width=height, 
                                    height=width,
                                    angle=angle))
                
            # add the ellipse
            p = PatchCollection(ells, cmap=cmap, edgecolor="gray", lw=.1)
            p.set_array(np.array(mean_w_logrels))
            ax.add_collection(p)
        
        # plot the holdout nodes
        if holdout_ids is not None:
            ax.scatter(grid[holdout_ids, 0],
                       grid[holdout_ids, 1],
                       edgecolors="red",
                       linewidth=node_lw,
                       s=1.2*node_size * np.sqrt(q),
                       color=node_color,
                       zorder=5)            
            

        if max_m is not None:
            ax.scatter(grid[:, 0],
                       grid[:, 1],
                       edgecolors="black",
                       s=((m>=max_m)*1.0)*20,
                       c=(m>=max_m)*1.0-1.0,
                       cmap=colors.ListedColormap(["white", "red"]),
                       zorder=2)

        if leg:
            # color scale params
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=clr.LogNorm(vmin=10**(-abs_max),
                                                        vmax=10**(abs_max)))
            sm._A = []

            # colorbar
            cbar = plt.colorbar(sm,
                                fraction=cbar_frac,
                                pad=cbar_pad,
                                orientation="horizontal")
            tick_locator = ticker.LogLocator(base=10, numticks=3)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.ax.tick_params(which="minor", length=0)

            # color bar title setting
            cbar.ax.set_title("centered log10 migration weight", loc="center")
            cbar.ax.set_title(cbar.ax.get_title(), fontsize=cbar_font_size)
                
        plt.axis("off")
        
