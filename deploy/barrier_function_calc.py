import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from qpsolvers import solve_qp
import xml.etree.ElementTree as ET

class controlBarrierFunction():

    def __init__(
        self,
        xml_path,
        exp_config
    ):

        # experiment configuration parameters
        robot_pos = exp_config["robot_pos"]
        target_body_origin = exp_config["target_pos"]
        shelf_pos = exp_config["shelf_pos"]
        moving_obs_pos = exp_config["moving_obs_pos"]
        self.moving_obs_cmd = np.array(exp_config["moving_obs_cmd"])
        self.alpha_static = exp_config["alpha_static"]
        self.alpha_moving = exp_config["alpha_moving"]
        self.gamma = exp_config["gamma"]
        self.x_weight = exp_config["x_weight"]
        self.y_weight = exp_config["y_weight"]
        self.theta_weight = exp_config["theta_weight"]
        self.slack_static = exp_config["slack_static_weight"]
        self.slack_moving = exp_config["slack_moving_weight"]
        self.slack_workspace = exp_config["slack_workspace_weight"]
        # data save file path
        self.pos_path = exp_config["pos_path"]
        self.cmd_path = exp_config["cmd_path"]
        self.col_path = exp_config["col_path"]

        # xml modifications and data extraction
        tree = ET.parse(xml_path)
        root = tree.getroot()
        target_mesh = root.find(".//mesh[@name='target_mesh']")
        target_mesh_path = target_mesh.get("file")
        target_mesh_path = "./" + target_mesh_path[target_mesh_path.find("custom_meshes"):]
        self.target_mesh = o3d.io.read_triangle_mesh(target_mesh_path)
        self.target_points = np.unique(np.asarray(self.target_mesh.vertices), axis=0)
        # update target position
        target_body = root.find(".//body[@name='target']")
        target_body.set("pos", target_body_origin)
        # update shelf position
        shelf_body = root.find(".//geom[@name='obs4']")
        shelf_body.set("pos", shelf_pos)
        tree.write(xml_path)
        print("XML Modified")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = robot_pos
        self.data.qpos[19:22] = moving_obs_pos  

        # represent target as a bounding box and express center in world coordinates
        min_x = np.min(self.target_points[:,0])
        max_x = np.max(self.target_points[:,0])
        min_y = np.min(self.target_points[:,1])
        max_y = np.max(self.target_points[:,1])
        min_z = np.min(self.target_points[:,2])
        max_z = np.max(self.target_points[:,2])
        self.target_pos = np.array([(max_x + min_x)/2, 
                                       (max_y + min_y)/2 ,
                                       (max_z + min_z)/2]) + np.fromstring(target_body_origin, dtype=np.float64, sep=' ')
        
        # convert workspace into convex hull for target reachability awareness
        pointcloud_path = "./pointcloud/workspace_point_cloud_filtered.npy"
        pointcloud = np.load(pointcloud_path)
        workspace_hull = ConvexHull(np.load(pointcloud_path))
        self.workspace_A = workspace_hull.equations[:, :3]
        self.workspace_b = workspace_hull.equations[:, 3]

        # calculated from offline manipulability study
        self.workspace_center = np.array([0.27320432, 0.00069574, 0.3312343])

        # initialize terms
        self.workspace_target_distance = 0.0
        self.h_static_obs = 0.0
        self.h_moving_obs = 0.0
        self.h_workspace = 0.0
        self.grad_h_static_obs = 0.0
        self.grad_h_moving_obs = 0.0
        self.grad_h_workspace = 0.0
        self.target_status = False


    def static_obs_calc(self, theta):
        model = self.model
        data = self.data

        distance_sensors = ["base_obs1",  "base_obs2", "base_obs3", "base_obs4"]
        normal_sensors = ["base_obs1_vec",  "base_obs2_vec", "base_obs3_vec", "base_obs4_vec"]

        distance_data = []
        for name in distance_sensors:
            distance_data.append(data.sensordata[model.sensor(name).adr])

        min_distance = np.min(distance_data)
        sensor_index = np.argmin(distance_data)
        normal_adr = model.sensor(normal_sensors[sensor_index]).adr[0]
        if min_distance > 0:
            grad = -data.sensordata[normal_adr:normal_adr+3]
        else:
            grad = data.sensordata[normal_adr:normal_adr+3]
        grad[2] = 0.0 # remove z component

        self.h_static_obs = min_distance

        # planar version of rotation matrix (pure yaw)
        R_world_to_body = np.array([
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                    ])
        self.grad_h_static_obs = R_world_to_body@grad

        return self.h_static_obs, self.grad_h_static_obs
    
    def moving_obs_calc(self, theta):
        model = self.model
        data = self.data

        distance = data.sensordata[model.sensor("base_obs5").adr]
        normal_adr = model.sensor("base_obs5_vec").adr[0]

        if distance > 0:
            grad = -data.sensordata[normal_adr:normal_adr+3]
        else:
            grad = data.sensordata[normal_adr:normal_adr+3]
        grad[2] = 0.0 # remove z component

        self.h_moving_obs = distance[0]

        # planar version of rotation matrix (pure yaw)
        R_world_to_body = np.array([
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                    ])
        self.grad_h_moving_obs = R_world_to_body@grad

        # time derivative 
        obs_vel = R_world_to_body @ self.data.qvel[18:]
        dh_dt = -self.grad_h_moving_obs @ obs_vel

        return self.h_moving_obs, self.grad_h_moving_obs, dh_dt

    def workspace_calc(self, theta):
        rel_target_pos = self.target_pos - self.data.qpos[:3]
        A = self.workspace_A.copy()
        b = self.workspace_b.copy()

        R_world_to_body = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
            ])
        grad_R_wrt_theta = np.array([
            [-np.sin(theta), np.cos(theta), 0],
            [-np.cos(theta), -np.sin(theta), 0],
            [0, 0, 0]
            ])
        rotated_target = R_world_to_body @ rel_target_pos
        self.workspace_target_distance = np.linalg.norm(rotated_target - self.workspace_center)
        self.h_workspace = self.workspace_target_distance
        
        grad_d_wrt_pos = -(rotated_target - self.workspace_center)/self.workspace_target_distance
        grad_d_wrt_theta = -grad_d_wrt_pos @ grad_R_wrt_theta @ rel_target_pos
        grad_d = np.concatenate((grad_d_wrt_pos[:2], [grad_d_wrt_theta]))
        self.grad_h_workspace = grad_d

        # print("Unscaled grad:", grad_d)
        signed_distances = A @ rotated_target + b
        if np.all(signed_distances<0):
            print("TARGET ACQUIRED")
            self.target_status = True

        return self.h_workspace, self.grad_h_workspace
    
    def qp_filter(self, u_d, theta):
        '''u_d is the policy output command (3,)'''

        h_static, grad_h_static = self.static_obs_calc(theta)
        h_moving, grad_h_moving, dh_dt = self.moving_obs_calc(theta)
        h_workspace, grad_h_workspace = self.workspace_calc(theta)
        grad_h_static = np.concatenate((grad_h_static, [1], [0], [0]))
        grad_h_moving = np.concatenate((grad_h_moving, [0], [1], [0]))
        grad_h_workspace = np.concatenate((grad_h_workspace, [0], [0], [-1]))
        
        # QP solver parameters
        P = np.diag([self.x_weight, self.y_weight, self.theta_weight, self.slack_static, self.slack_moving, self.slack_workspace])
        q = -P @ np.concatenate((u_d, [0.0], [0.0], [0.0]))
        G = np.vstack((-grad_h_static, -grad_h_moving, grad_h_workspace))
        h = np.array([[self.alpha_static * h_static],
                      [self.alpha_moving * h_moving + dh_dt],
                      [-self.gamma * h_workspace]])
        lb = 1.0 * np.array([-1,-1,-1, 0, 0, 0])
        ub = 1.0 * np.array([1, 1, 1, 10, 10, 10])
        # print("Gu <", h)
        solution = solve_qp(P, q, G, h, ub=ub, lb=lb, solver="cvxopt")
        u = np.round(solution[:3], 2)
        static_slack = solution[3]
        moving_slack = solution[4]
        workspace_slack = solution[5]

        # print("h static:", h_static)
        # print("h moving:", h_moving)
        # print("h workspace:", h_workspace)
        # print("static obstacle grad:", grad_h_static)
        # print("moving obstacle grad:", grad_h_moving)
        # print("workspace grad:", grad_h_workspace)
        # print("==================================================================")

        return u, static_slack, moving_slack, workspace_slack

        