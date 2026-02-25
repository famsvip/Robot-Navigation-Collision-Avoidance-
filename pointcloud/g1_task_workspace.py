import mujoco
from mujoco import viewer
import numpy as np
import time
import open3d as o3d

# Load the model (using the 23dof model, since the motion policy doesn't have arm dof)
model = mujoco.MjModel.from_xml_path("../unitree_rl_gym/resources/robots/g1_description/g1_23dof_rev_1_0.xml")
data = mujoco.MjData(model)

joint_names = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
               'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']

joint_limits = [[model.jnt_range[model.joint(name).id][0], model.jnt_range[model.joint(name).id][1]] for name in joint_names]

lpalm_id = model.site('left_palm').id
rpalm_id = model.site('right_palm').id

v = mujoco.viewer.launch_passive(model, data)

ee_points = []
N = 100000 # number of samples
for n in range(N):
    for name in joint_names:
        # get the joint limit range
        joint_id = model.joint(name).id
        low = model.jnt_range[joint_id][0]
        high = model.jnt_range[joint_id][1]
        # update qpos for sampled joint positions
        joint_q = model.jnt_qposadr[joint_id]
        data.qpos[joint_q] = np.random.uniform(low, high)

    mujoco.mj_forward(model, data)
    
    # ignore joint configurations that lead to self-collision
    if data.ncon > 0:
        continue

    root_pos = data.qpos[:3].copy()
    lpalm_pos = data.site_xpos[lpalm_id].copy()
    rpalm_pos = data.site_xpos[rpalm_id].copy()

    lpalm_pos -= root_pos
    rpalm_pos -= root_pos

    # filter for front cone in workspace
    lpalm_angle = np.arctan2(lpalm_pos[0], lpalm_pos[1])
    rpalm_angle = np.arctan2(rpalm_pos[0], rpalm_pos[1])
    if lpalm_pos[0] > 0 and lpalm_angle > np.radians(47) and lpalm_angle < np.radians(133):
        ee_points.append(lpalm_pos)
    if rpalm_pos[0] > 0 and rpalm_angle > np.radians(47) and rpalm_angle < np.radians(133):
        ee_points.append(rpalm_pos)

    v.sync()
    time.sleep(0.01)

v.close()
ee_points = np.array(ee_points)
print(ee_points.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ee_points)
voxel_size = 0.01
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
o3d.visualization.draw_geometries([voxel_grid])

np.save("./workspace_point_cloud_filtered", ee_points)


# ### Check max forward position
# v = mujoco.viewer.launch_passive(model, data)

# shoulder_joint = -1.57
# joint_id = model.joint("left_shoulder_pitch_joint").id
# joint_q = model.jnt_qposadr[joint_id]
# data.qpos[joint_q] = shoulder_joint

# elbow_joint = 1.57
# joint_id = model.joint("left_elbow_joint").id
# joint_q = model.jnt_qposadr[joint_id]
# data.qpos[joint_q] = elbow_joint

# mujoco.mj_forward(model, data)

# root_pos = data.qpos[:3].copy()
# lpalm_pos = data.site_xpos[lpalm_id].copy()


# lpalm_pos -= root_pos
# print(lpalm_pos)

# while True:
#     v.sync()
