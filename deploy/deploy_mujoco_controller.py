import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import threading
from pynput import keyboard
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

def contact_check(m, d, obs_frc):
    contact_indices = np.argwhere(d.contact.geom1 == 4).flatten()
    if contact_indices.size != 0:
        for i in contact_indices:
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            obs_frc.append(np.linalg.norm(force[:3]))
        
        print("Shelf contact detected")
        print("obs_frc:", obs_frc)
    return None

# Global command variable
cmd = None
cmd_lock = threading.Lock()

def on_press(key):
    """Background keyboard listener callback"""
    global cmd
    
    try:
        key_char = key.char
    except AttributeError:
        return
    
    with cmd_lock:
        if cmd is None:
            return
            
        step = 0.1
        
        if key_char == 'w':
            cmd[0] = min(cmd[0] + step, 1.0)
            print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]", flush=True)
        elif key_char == 's':
            cmd[0] = max(cmd[0] - step, -1.0)
            print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]", flush=True)
        elif key_char == 'a':
            cmd[1] = min(cmd[1] + step, 1.0)
            print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]", flush=True)
        elif key_char == 'd':
            cmd[1] = max(cmd[1] - step, -1.0)
            print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]", flush=True)
        elif key_char == 'q':
            cmd[2] = min(cmd[2] + step, 1.0)
            print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]", flush=True)
        elif key_char == 'e':
            cmd[2] = max(cmd[2] - step, -1.0)
            print(f"CMD: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}]", flush=True)
        elif key_char == ' ':
            cmd[:] = 0.0
            print(f"CMD: STOP", flush=True)
        elif key_char == 'r':
            cmd[:] = [0.0, 0.0, 0.0]
            print(f"CMD: RESET", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_config", type=str, help="config file name in the config folder")
    parser.add_argument("--plot", action="store_true", default=False, help="turn on or off plotting")
    args = parser.parse_args()
    exp_config_file = args.exp_config
    
    with open(f"../unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml", "r") as f:
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

    # Print controls
    print("\n" + "="*60)
    print("KEYBOARD CONTROLS (background listener):")
    print("  W/S     - Increase/Decrease forward velocity")
    print("  A/D     - Increase/Decrease lateral velocity") 
    print("  Q/E     - Increase/Decrease yaw rate")
    print("  SPACE   - Stop all motion")
    print("  R       - Reset to zero")
    print("="*60 + "\n")

    # Start background keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("✓ Keyboard listener started in background")

    # Define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load qp filter
    with open(f"./exp_config/{exp_config_file}", "r") as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
        cbf = controlBarrierFunction(xml_path, exp_config)
    moving_obs_cmd = cbf.moving_obs_cmd

    # Load robot model
    m = cbf.model
    d = cbf.data
    m.opt.timestep = simulation_dt

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"✓ Policy loaded from {policy_path}\n")

    # Store data
    root_pos = []
    mod_cmds = []
    obs_frc = []
    constraint_values = []
    max_recorded_step = 2500

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            
            start = time.time()
            buffer_index = 0
            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()
                tau = pd_control(target_dof_pos, d.qpos[7:19], kps, np.zeros_like(kds), d.qvel[6:18], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

                yaw_angle = root_yaw(d.qpos[3:7])

                 # Get current cmd value (thread-safe)
                with cmd_lock:
                    current_cmd = cmd.copy()
                    current_cmd = [1.0, 0.0, 0.0]

                modified_cmd, static_slack, moving_slack = cbf.qp_filter(current_cmd, yaw_angle)
                if counter < max_recorded_step and cbf.target_status != True:
                    root_pos.append(np.concatenate((d.qpos[:2], [yaw_angle])))
                    mod_cmds.append(modified_cmd)
                    print(d.qvel[:2])
                    terminal_vel = np.linalg.norm(d.qvel[:2])
                    constraint_values.append([cbf.h_static_obs, cbf.h_static_obs, cbf.h_moving_obs, np.linalg.norm(cbf.grad_h_static_obs), moving_slack, static_slack, terminal_vel])
                    contact_check(m, d, obs_frc)
                    print("Recording")

                print("Simulation count:", counter, "|| original cmd:", current_cmd, "|| modified cmd:", modified_cmd )
                print("static h:", cbf.h_static_obs)
                print("static slack:", static_slack)
                print("grad_h_static:", cbf.grad_h_static_obs)
              
                counter += 1
                if counter % control_decimation == 0:
                    
                    # Create observation
                    qj = d.qpos[7:19]
                    dqj = d.qvel[6:18]
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
                    obs[6:9] = modified_cmd * cmd_scale
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
                
                # Update moving obstacles
                if np.abs(d.qpos[20])>3:
                    moving_obs_cmd *= -1
                d.qvel[18:21] = cbf.moving_obs_cmd
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        listener.stop()
        print("Keyboard listener stopped")
        # np.save(cbf.pos_path, np.array(root_pos))
        # np.save(cbf.cmd_path, np.array(mod_cmds))
        # np.save(cbf.col_path, np.array(obs_frc))
        # np.save(cbf.val_path, np.array(constraint_values))
        print("Data saved")
        print(cbf.target_status)