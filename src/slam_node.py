#!/usr/bin/env python3
import open3d as o3d
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from scipy.io import savemat
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

VOXEL_SIZE = 15.0
VISUALIZE = False
WORLD_FRAME_START = False

# Add the provided code with the FrameToFrameRegistration function here

def FrameToFrameRegistration(A_pcd, B_pcd, voxel_size):
    print("-----------------------------")
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
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp = icp_sol.transformation

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
    # o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])
    return T_icp

class SLAMNode:
    def __init__(self):
        self.source_pcd = None
        self.target_pcd = None
        self.source_received = False
        self.target_received = False
        self.buffer = []
        self.buffer_threshold = 6

        self.source_sub = rospy.Subscriber('source_pointcloud', PointCloud2, self.source_callback)
        # self.target_sub = rospy.Subscriber('target_pointcloud', PointCloud2, self.target_callback)

    def source_callback(self, msg):
        print("enter source callback")
        self.source_pcd = self.point_cloud2_to_open3d(msg)
        self.buffer.append(self.source_pcd)
        if len(self.buffer) >= self.buffer_threshold:
            self.process_point_clouds()
            self.buffer.clear()
            print("clear the frame buffer, waiting for next round")
        # self.source_received = True
        # self.process_point_clouds()

    def target_callback(self, msg):
        print("enter target callback")
        self.target_pcd = self.point_cloud2_to_open3d(msg)
        self.target_received = True
        self.process_point_clouds()

    def point_cloud2_to_open3d(self, msg):
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(list(points))
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        return pcd

    def process_point_clouds(self):
        print("enter process_point_cloud function")
        # print(f"source_received {self.source_received}")
        # print(f"target_received {self.target_received}")
        # if self.source_received and self.target_received:
            # print("satisfy condition")
        # Call the FrameToFrameRegistration function with the received point clouds
        refined_poses = []
        global_point_cloud = o3d.geometry.PointCloud()
        for i in range(len(self.buffer)-1):
            source_id = i
            target_id = i+1
            T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
            refined_poses.append(T_icp)
        if WORLD_FRAME_START:
            for i in range(1,self.buffer_threshold):
                for j in range(i-1,-1,-1):
                    self.buffer[i].transform(np.linalg.inv(refined_poses[j]))
                global_point_cloud += self.buffer[i]
            global_point_cloud += self.buffer[0]
        else:
            for i in range(self.buffer_threshold-1):
                for j in range(i,self.buffer_threshold-1):
                    self.buffer[i].transform(refined_poses[j])
                global_point_cloud += self.buffer[i]
            global_point_cloud += self.buffer[len(self.buffer)-1]
        o3d.visualization.draw_geometries([global_point_cloud])

def main():
    rospy.init_node('slam_node')
    slam_node = SLAMNode()
    rospy.spin()

if __name__ == '__main__':
    main()