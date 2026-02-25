import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import threading

import navigation_env_train
import navigation_env
from stable_baselines3 import PPO
from barrier_function_calc import controlBarrierFunction

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def root_yaw(quat):
    """Given the quaternion of the root, calculate the yaw angle"""

    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    num = 2 * (q0*q3 + q1*q2)
    denom = q0**2 + q1**2 - q2**2 - q3**2
    yaw_angle = np.arctan2(num, denom)

    return yaw_angle

# Global command variable
cmd = None
cmd_lock = threading.Lock()

# Navigation policy setup
nav_policy_path = "../warehouse/nn/nav/2026-02-11 13:27:43.483356/best_model"
env = navigation_env_train.make_env()
nav_model = PPO.load(nav_policy_path, env=env)
obs, _ = env.reset()
nav_class = navigation_env.NavWorldEnv(size=10)
cmd_dict = nav_class.action_to_vel
print("Navigation policy loaded")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(f"../unitree_rl_gym/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = "../unitree_rl_gym/deploy/pre_train/g1/motion.pt"
        xml_path = "../unitree_rl_gym/resources/robots/g1_description/warehouse_scene.xml"

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        # Initialize global cmd
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"✓ Policy loaded from {policy_path}\n")
    # Load qp filter
    pointcloud_path = "./workspace_point_cloud_filtered.npy"
    target_path = "./target_sphere.stl"
    cbf = controlBarrierFunction(m, d, pointcloud_path, xml_path)
    target_pos = cbf.target_pos
    print("Target Pos:", target_pos)

    standing = False    

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:

            d.qpos[:2] = [5.5, 0]

            start = time.time()
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

                counter += 1
                if counter % control_decimation == 0:
                    # Construct obs for nav_policy
                    yaw_angle = root_yaw(d.qpos[3:7])
                    agent_pos = np.array([d.qpos[0], d.qpos[1], yaw_angle])
                    
                    rel_distance_world = target_pos[:2] - agent_pos[:2]
                    distance = np.array([np.linalg.norm(rel_distance_world)])

                     # relative distance in body frame
                    rel_dx = rel_distance_world[0] * np.cos(yaw_angle) + rel_distance_world[1] * np.sin(yaw_angle)
                    rel_dy = -rel_distance_world[0] * np.sin(yaw_angle) + rel_distance_world[1] * np.cos(yaw_angle)
                    rel_distance_body = np.array([rel_dx, rel_dy], dtype=np.float64)

                    # relative yaw error
                    desired_yaw = np.arctan2(rel_distance_world[1], rel_distance_world[0])
                    rel_yaw_error = desired_yaw - yaw_angle
                    rel_yaw_error = (rel_yaw_error + np.pi) % (2*np.pi) - np.pi
                    rel_yaw_error = np.array([rel_yaw_error], dtype=np.float64)

                    nav_obs = {
                        "agent": agent_pos,
                        "target": target_pos[:2],
                        "distance": distance,
                        "relative_distance_body": rel_distance_body,
                        "relative_yaw_error": rel_yaw_error
                    }
                    
                    nav_action, _ = nav_model.predict(nav_obs, deterministic=True)
                    current_cmd = cmd_dict[int(nav_action.item())]
                    
                    distance_sensors = ["base_obs1",  "base_obs2", "base_obs3", "base_obs4"]

                    original_cmd = current_cmd
                    cbf_check = False
                    distance_measurements = []
                    for name in distance_sensors:
                        distance_measurements.append(d.sensordata[m.sensor(name).id])
                        if d.sensordata[m.sensor(name).id] < 1.0:
                            cbf_check = True
                    if cbf_check:
                        current_cmd = cbf.qp_filter(current_cmd, yaw_angle)
                    
                    print("original cmd:", original_cmd, "|| modified cmd:", current_cmd )
                   
                    # Create observation
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale

                    period = 0.8
                    count = counter * simulation_dt
                    phase = count % period / period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)

                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = current_cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    
                    # Policy inference
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    target_dof_pos = action * action_scale + default_angles
                    
                viewer.sync()

                # Time keeping
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")



# '''Math check'''
# import navigation_env_train as nav
# from navigation_env import NavWorldEnv

# # Test specific action sequences to verify behavior
# env = nav.make_env()
# obs, info = env.reset(seed=42)  # Use seed for reproducible testing

# print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

# # Test each action type
# nav = NavWorldEnv(10)
# action_to_vel = nav.action_to_vel
# print(action_to_vel)

# for action in action_to_vel:
#     old_pos = obs['agent'].copy()
#     obs, reward, terminated, truncated, info = env.step(action)
#     new_pos = obs['agent']
#     print(f"Action {action_to_vel[action]}: {old_pos} -> {new_pos}, reward={reward}")
