import open3d as o3d
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from scipy.io import savemat

VOXEL_SIZE = 15.0
VISUALIZE = False
WORLD_FRAME_START = False

def FrameToFrameRegistration(A_pcd, B_pcd, voxel_size):
    A_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
    B_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red

    # voxel downsample both clouds
    # A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 

    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd,VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd,VOXEL_SIZE)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]
    print(f'FPFH generates {num_corrs} putative correspondences.')

    # visualize the point clouds together with feature correspondences
    points = np.concatenate((A_corr.T,B_corr.T),axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

    # robust global registration using TEASER++
    NOISE_BOUND = VOXEL_SIZE
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)

    # Visualize the registration results
    A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
    # o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])

    # local refinement using ICP
    icp_sol = o3d.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, T_teaser,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp = icp_sol.transformation

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
    # o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])
    return T_icp

if __name__ == "__main__":
    print("==================================================")
    print("        Frame-To-Frame registration example      ")
    print("==================================================")
    global_point_cloud = o3d.geometry.PointCloud()
    refined_poses = []
    pcds_down = []
    origin_pcds_down = []
    num_of_scan = 6
    for i in range(num_of_scan):
        pcd_raw = o3d.io.read_point_cloud('./data/plane%d.pcd' % (i+1))
        pcd_down = pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
        origin_pcd_down = copy.deepcopy(pcd_down)
        pcds_down.append(pcd_down)
        origin_pcds_down.append(origin_pcd_down)

    for i in range(num_of_scan-1):
        source_id = i
        target_id = i+1
        T_icp = FrameToFrameRegistration(pcds_down[source_id],pcds_down[target_id],VOXEL_SIZE)
        refined_poses.append(T_icp)
    refine_poses_dict = {f'pose{i+1}': pose for i,pose in enumerate(refined_poses)}
    savemat('Trans_plane.mat', refine_poses_dict)
    if WORLD_FRAME_START:
        for i in range(1,num_of_scan):
            for j in range(i-1,-1,-1):
                origin_pcds_down[i].transform(np.linalg.inv(refined_poses[j]))
            global_point_cloud += origin_pcds_down[i]
        global_point_cloud += origin_pcds_down[0]
    else:
        for i in range(num_of_scan-1):
            for j in range(i,num_of_scan-1):
                origin_pcds_down[i].transform(refined_poses[j])
            global_point_cloud += origin_pcds_down[i]
        global_point_cloud += origin_pcds_down[len(origin_pcds_down)-1]
    o3d.visualization.draw_geometries([global_point_cloud])
