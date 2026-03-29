import mujoco
from mujoco import viewer
import numpy as np
import time
import open3d as o3d
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--generate", action="store_true", default=False, help="turn on or off plotting")
parser.add_argument("--load", action="store_true", default=False, help="turn on or off plotting")
args = parser.parse_args()

if args.generate:
    # Load the model (using the 23dof model, since the motion policy doesn't have arm dof)
    model = mujoco.MjModel.from_xml_path("../unitree_rl_gym/resources/robots/g1_description/g1_23dof_rev_1_0.xml")
    data = mujoco.MjData(model)

    joint_names = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
    joint_limits = [[model.jnt_range[model.joint(name).id][0], model.jnt_range[model.joint(name).id][1]] for name in joint_names]

    lpalm_site_id = model.site('left_palm').id
    rpalm_site_id = model.site('right_palm').id
    lpalm_body_id = model.body('left_wrist_roll_rubber_hand').id
    rpalm_body_id = model.body('right_wrist_roll_rubber_hand').id

    l_Jp = np.zeros((3,model.nv))
    r_Jp = np.zeros((3,model.nv))

    # for i in range(model.njnt):
    #     print(model.joint(i).name, model.joint(i).dofadr)

    v = mujoco.viewer.launch_passive(model, data)

    N = 100000 # number of samples
    ee_points = []
    manipulability_scores = []
    p_refs = np.empty((1,3), dtype=np.float64)
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
        lpalm_pos = data.site_xpos[lpalm_site_id].copy()
        rpalm_pos = data.site_xpos[rpalm_site_id].copy()

        mujoco.mj_jac(model, data, l_Jp, None, lpalm_pos, lpalm_body_id)
        l_Jp_sliced = l_Jp[:, 19:24]
        l_manipulability = np.sqrt(np.linalg.det(l_Jp_sliced @ l_Jp_sliced.T))
        mujoco.mj_jac(model, data, r_Jp, None, rpalm_pos, rpalm_body_id)
        r_Jp_sliced = r_Jp[:, 24:]
        r_manipulability = np.sqrt(np.linalg.det(r_Jp_sliced @ r_Jp_sliced.T))

        lpalm_pos -= root_pos
        rpalm_pos -= root_pos

        # filter for front cone in workspace
        lpalm_angle = np.arctan2(lpalm_pos[0], lpalm_pos[1])
        rpalm_angle = np.arctan2(rpalm_pos[0], rpalm_pos[1])
        if lpalm_pos[0] > 0 and lpalm_angle > np.radians(47) and lpalm_angle < np.radians(133):
            ee_points.append(lpalm_pos)
            manipulability_scores.append(l_manipulability)
        if rpalm_pos[0] > 0 and rpalm_angle > np.radians(47) and rpalm_angle < np.radians(133):
            ee_points.append(rpalm_pos)
            manipulability_scores.append(r_manipulability)

        # calculate the weighted reference point
        points = np.array(ee_points)
        scores = np.array(manipulability_scores)
        weighted_ref = np.sum(points * scores[:, None], axis=0)/np.sum(scores)
        p_refs = np.vstack((p_refs, [weighted_ref]))
        print("Count:", n, "||", weighted_ref)

        v.sync()
        time.sleep(0.01)

    v.close()
    np.save("./pointcloud/workspace_pointcloud_manipulate", np.array(ee_points))
    np.save("./pointcloud/workspace_manipulate_scores", np.array(manipulability_scores))
    np.save("./pointcloud/workspace_weighted_point", p_refs)
    print(np.array(ee_points).shape)
    print(np.array(manipulability_scores).shape)

print("Done")

if args.load:
    points = np.load("./pointcloud/workspace_pointcloud_manipulate.npy")
    scores = np.load("./pointcloud/workspace_manipulate_scores.npy")

    # threshold = 0.0175
    # score_filter = scores > threshold
    # print("Average point within threshold:", np.sum(points[score_filter], axis=0)/np.sum(score_filter))
    print("Average point within threshold:", np.sum(points * scores[:, None], axis=0)/np.sum(scores))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:,0], points[:,1], points[:,2], c=scores, cmap='viridis', marker='o')
    fig.colorbar(sc, ax=ax, pad=0.1, label='Manipulability Score')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
