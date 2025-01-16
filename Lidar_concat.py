import numpy as np
import glob
import os
import mmcv
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import pickle

def read_pcd_bin(file_path):
    point_cloud = np.fromfile(file_path, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 5))
    return point_cloud


# Read the PCD file
def read_all_pcd_files(directory_path):
    pcd_files = glob.glob(os.path.join(directory_path, '*.pcd.bin'))
    point_clouds = []

    for file_path in pcd_files:
        points = read_pcd_bin(file_path)
        point_clouds.append(points)

    return point_clouds


# Function to apply transformation
def transform_points(points, R, T):
    # Convert points to homogeneous coordinates
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply rotation and translation
    points_transformed = (transformation_matrix @ points_homogeneous.T).T
    # Return only the x, y, z coordinates
    return points_transformed[:, :3]


# Directory containing the .pcd.bin files
# directory_path_infra = '/media/jingxiong/HaoranSSD/V2X-SIM/samples/LIDAR_TOP_id_0'
# directory_path_vehicle = '/media/jingxiong/HaoranSSD/V2X-SIM/samples/LIDAR_TOP_id_1'

# # Read all PCD files in the directory
# all_point_clouds_infra = read_all_pcd_files(directory_path_infra)
# all_point_clouds_vehicle = read_all_pcd_files(directory_path_vehicle)

# Do infra->vehicle transformation
transform_dict = {}
nusc = NuScenes(version='v1.0-trainval', dataroot='/scratch/jmeng18/V2X-SIM', verbose=True)
for sample in mmcv.track_iter_progress(nusc.sample):
    # Get lidar_vehicle transformation matrix
    lidar_token = sample['data']['LIDAR_TOP_id_1']
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP_id_1'])
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    l2e_r = cs_record['rotation']
    l2e_t = cs_record['translation']
    e2g_r = pose_record['rotation']
    e2g_t = pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # Get lidar_infra transformation matrix
    lidar_token_infra = sample['data']['LIDAR_TOP_id_0']
    sd_rec_infra = nusc.get('sample_data', sample['data']['LIDAR_TOP_id_0'])
    cs_record_infra = nusc.get('calibrated_sensor',
                         sd_rec_infra['calibrated_sensor_token'])
    pose_record_infra = nusc.get('ego_pose', sd_rec_infra['ego_pose_token'])
    l2e_r_s = cs_record_infra['rotation']
    l2e_t_s = cs_record_infra['translation']
    e2g_r_s = pose_record_infra['rotation']
    e2g_t_s = pose_record_infra['translation']
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat= Quaternion(e2g_r_s).rotation_matrix

    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T

    # Do transformation to infra_lidar
    # point_clouds_infra = read_pcd_bin('/media/jingxiong/HaoranSSD/V2X-SIM/'+sd_rec_infra['filename'])
    # point_cloud = read_pcd_bin('/media/jingxiong/HaoranSSD/V2X-SIM/'+sd_rec['filename'])
    #
    # point_cloud_infra_geo = point_clouds_infra[:,:3]
    # points_infra_transformed = transform_points(point_cloud_infra_geo, R, T)
    #
    # point_clouds_infra[:, :3] = points_infra_transformed

    # Create a dict contains infra_lidar->vehicle_lidar
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R.T
    transformation_matrix[:3, 3] = T
    transform_dict['/scratch/jmeng18/V2X-SIM/'+sd_rec_infra['filename']] = transformation_matrix


with open('lidar_transform_dict.pkl', 'wb') as file:
    pickle.dump(transform_dict, file)

    # Concat two point cloud sets
    # point_cloud_concat = np.vstack((point_cloud, point_clouds_infra))
    # cloud_shape = point_cloud_concat.shape
    # reshaped_point_cloud = point_cloud_concat.reshape(cloud_shape[0]*cloud_shape[1])

    # output_file_path = '/media/jingxiong/HaoranSSD/V2X-SIM/lidar_concat/' + sd_rec['filename']
    # output_dir = os.path.dirname(output_file_path)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # reshaped_point_cloud.astype(np.float32).tofile(output_file_path)


