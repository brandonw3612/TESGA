import numpy as np
import open3d as o3d

from pc_skeletor.lbc import LBC


if __name__ == "__main__":
    cloud_path = './data/bag_6.ply'
    pcd = o3d.io.read_point_cloud(cloud_path)

    # Laplacian-based Contraction
    lbc = LBC(point_cloud=pcd, init_contraction=3.,
              init_attraction=0.6,
              max_contraction=2048,
              max_attraction=1024,
              down_sample=0.02)
    downsampled = lbc.get_points()
    # export downsampled point cloud
    o3d.io.write_point_cloud('./output/downsampled.ply', o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downsampled)))
    contracted = lbc.extract_skeleton()
    o3d.io.write_point_cloud('./output/contracted.ply', o3d.geometry.PointCloud(o3d.utility.Vector3dVector(contracted)))
    lbc.extract_topology()

    # lbc.show_graph(lbc.skeleton_graph)
    # lbc.show_graph(lbc.topology_graph)
    # lbc.visualize()
    # lbc.export_results('./output')
# lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=300, output='./output_1')