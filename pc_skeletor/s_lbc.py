"""
Original source: https://github.com/meyerls/pc-skeletor/blob/main/pc_skeletor/laplacian.py
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import open3d.visualization as o3d
import scipy.sparse.linalg as sla
from scipy import sparse

from pc_skeletor.lbc_base import LaplacianBasedContractionBase


class SLBC(LaplacianBasedContractionBase):
    """
    Semantic Laplacian-Based Contraction (S-LBC)
    ---------------------------------

    Our semantic skeletonization algorithm based on Laplacian-Based Contraction.

    Paper: https://arxiv.org/abs/2304.04708

    """

    def __init__(self,
                 point_cloud: Union[str, dict],
                 semantic_weighting: float = 10.,
                 init_contraction: float = 1.,
                 init_attraction: float = 0.5,
                 max_contraction: int = 2048,
                 max_attraction: int = 1024,
                 step_wise_contraction_amplification: Union[float, str] = 'auto',
                 termination_ratio: float = 0.003,
                 max_iteration_steps: int = 20,
                 down_sample: float = -1.,
                 filter_nb_neighbors: int = 20,
                 filter_std_ratio: float = 2.0,
                 debug: bool = False,
                 verbose: bool = False):
        super().__init__(self.__class__.__name__,
                         point_cloud,
                         init_contraction,
                         init_attraction,
                         max_contraction,
                         max_attraction,
                         step_wise_contraction_amplification,
                         termination_ratio,
                         max_iteration_steps,
                         debug,
                         verbose,
                         self.__least_squares_sparse)

        self.semantic_weighting = semantic_weighting
        # Down sampling point cloud for faster contraction.
        if down_sample != -1:
            self.trunk = self.trunk.voxel_down_sample(down_sample)
            self.branches = self.branches.voxel_down_sample(down_sample)

        # Filter point cloud as outliers might distort the skeletonization algorithm
        if filter_nb_neighbors and filter_std_ratio:
            self.trunk, _ = self.trunk.remove_statistical_outlier(nb_neighbors=filter_nb_neighbors,
                                                                  std_ratio=filter_std_ratio)
            self.branches, _ = self.branches.remove_statistical_outlier(nb_neighbors=filter_nb_neighbors,
                                                                        std_ratio=filter_std_ratio)
        self.pcd = self.trunk + self.branches

        if self.debug:
            o3d.visualization.draw_geometries([self.pcd], window_name="Default Point Cloud")

    def __least_squares_sparse(self, pcd_points, L, laplacian_weighting, positional_weighting):
        """
        Perform least squares sparse solving for the Semantic Laplacian-Based Contraction (S-LBC).

        Args:
            pcd_points: The input point cloud points.
            L: The Laplacian matrix.
            laplacian_weighting: The Laplacian weighting matrix.
            positional_weighting: The positional weighting matrix.

        Returns:
            The contracted point cloud.
        """
        # Define Weights
        WL = sparse.diags(laplacian_weighting)  # I * laplacian_weighting
        WH = sparse.diags(positional_weighting)

        num_amplification_points = np.asarray(self.trunk.points).shape[0]
        multiplier = self.semantic_weighting

        # Very complicated way to create a sparse matrix with the upper matrix to have the correct weights for the trunk.
        S = sparse.csc_matrix.copy(L)
        rows = S.nonzero()[0]
        cols = S.nonzero()[1]
        pos = np.vstack([rows, cols]).T
        connection_pos = np.where(pos[:, 1] > num_amplification_points)[0]
        connection_pos_idx = np.unique(pos[connection_pos][:, 0])
        mask = np.ones(pcd_points.shape[0], np.bool_)

        mask[connection_pos_idx] = 0
        num_valid = np.arange(0, pcd_points.shape[0])[mask]
        S[rows, cols] = 1

        # ToDo: Speed up!
        for i in num_valid:
            S[i, L[i].nonzero()[1]] = multiplier

        WL_L = (WL @ L)
        WL_L = WL_L.multiply(S)

        if self.debug:
            plt.figure(figsize=(10, 10))

            plt.spy(L, ms=1)
            plt.title('Laplacian Matrix', fontsize=40)
            plt.show()

        A = sparse.vstack([WL_L, WH]).tocsc()
        b = np.vstack([np.zeros((pcd_points.shape[0], 3)), WH.dot(pcd_points)])

        A_new = A.T @ A

        x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
        y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
        z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec='COLAMD')

        ret = np.vstack([x, y, z]).T

        if (np.isnan(ret)).all():
            logging.warning('Matrix is exactly singular. Stopping Contraction.')
            ret = pcd_points

        if self.debug:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            mean_curvature_flow = WL_L @ pcd_points
            # Scale normals for visualization
            pcd.normals = o3d.utility.Vector3dVector(mean_curvature_flow / 5)

            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        return ret