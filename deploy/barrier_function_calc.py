import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from qpsolvers import solve_qp
import xml.etree.ElementTree as ET
import open3d as o3d

class controlBarrierFunction():

    def __init__(
        self,
        model,
        data,
        xml_path
    ):
        self.model = model
        self.data = data

        # extract mesh information from xml
        tree = ET.parse(xml_path)
        root = tree.getroot()
        target_mesh = root.find(".//mesh[@name='target_mesh']")
        target_mesh_path = target_mesh.get("file")
        target_mesh_path = "./" + target_mesh_path[target_mesh_path.find("custom_meshes"):]
        self.target_mesh = o3d.io.read_triangle_mesh(target_mesh_path)
        self.target_points = np.unique(np.asarray(self.target_mesh.vertices), axis=0)
        target_body = root.find(".//body[@name='target']")
        target_body_origin = target_body.get("pos")
        target_body_origin = np.fromstring(target_body_origin, sep=' ')

        # represent target as a bounding box and express center in world coordinates
        min_x = np.min(self.target_points[:,0])
        max_x = np.max(self.target_points[:,0])
        min_y = np.min(self.target_points[:,1])
        max_y = np.max(self.target_points[:,1])
        min_z = np.min(self.target_points[:,2])
        max_z = np.max(self.target_points[:,2])
        self.target_pos = np.array([(max_x + min_x)/2, 
                                       (max_y + min_y)/2 ,
                                       (max_z + min_z)/2]) + target_body_origin
        
        # convert workspace into convex hull
        pointcloud_path = "./pointcloud/workspace_point_cloud_filtered.npy"
        pointcloud = np.load(pointcloud_path)
        self.workspace_center = np.sum(pointcloud, 0)/pointcloud.shape[0]

        # workspace_mesh = o3d.io.read_triangle_mesh("./workspace_mesh.stl")
        # workspace_mesh.compute_vertex_normals()
        # self.scene = o3d.t.geometry.RaycastingScene()
        # self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(workspace_mesh))

        workspace_hull = ConvexHull(np.load(pointcloud_path))
        self.workspace_A = workspace_hull.equations[:, :3]
        self.workspace_b = workspace_hull.equations[:, 3]

        # initialize terms
        self.workspace_target_distance = 0.0
        self.h_obs = 0.0
        self.h_workspace = 0.0
        self.grad_h_obs = 0.0
        self.grad_h_workspace = 0.0


    def obstacle_calc(self, theta):
        model = self.model
        data = self.data

        distance_sensors = ["base_obs1",  "base_obs2", "base_obs3", "base_obs4", "base_obs5"]
        normal_sensors = ["base_obs1_vec",  "base_obs2_vec", "base_obs3_vec", "base_obs4_vec", "base_obs5_vec"]

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

        self.h_obs = min_distance

        # planar version of rotation matrix (pure yaw)
        R_world_to_body = np.array([
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                    ])
        self.grad_h_obs = R_world_to_body@grad

        # time derivative 
        if sensor_index == 4:
            obs_vel = R_world_to_body @ self.data.qvel[18:]
        else:
            obs_vel = np.array([0.0, 0.0, 0.0])
        dh_dt = -self.grad_h_obs @ obs_vel

        return self.h_obs, self.grad_h_obs, dh_dt

    def workspace_calc(self, alpha, beta, gamma, theta):
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
        sigmoid =  1/(1+np.exp(beta*(self.workspace_target_distance - gamma)))
        self.h_workspace = alpha * sigmoid
        print("Rotated Target:", rotated_target)
        print("Workspace Center:", self.workspace_center)
        print("Vector from center to target:", rotated_target - self.workspace_center)
        
        grad_d_wrt_pos = -(rotated_target - self.workspace_center)/self.workspace_target_distance
        grad_d_wrt_theta = -grad_d_wrt_pos @ grad_R_wrt_theta @ rel_target_pos
        grad_d = np.concatenate((grad_d_wrt_pos[:2], [grad_d_wrt_theta]))
        self.grad_h_workspace = -alpha * beta * sigmoid * (1-sigmoid) * grad_d

        print("Unscaled grad:", grad_d)
        signed_distances = A @ rotated_target + b
        if np.all(signed_distances<0):
            print("TARGET ACQUIRED")

        return self.h_workspace, self.grad_h_workspace
    
    def composite_calc(self, theta):
        h_obs, grad_h_obs, h_dot = self.obstacle_calc(theta)
        h_work, grad_h_work = self.workspace_calc(0.5, 2, 0.5, theta)

        h_composite = h_obs + h_work
        grad_h = grad_h_obs + grad_h_work

        print("h obstacle:", h_obs)
        print("h workspace:", h_work)
        print("obstacle grad:", grad_h_obs)
        print("workspace grad:", grad_h_work)

        return np.array([h_composite]), grad_h, h_dot
    
    def qp_filter(self, u_d, theta):
        '''u_d is the policy output command (3,)'''

        alpha = 0.1
        h_comp, h_grad, h_dot = self.composite_calc(theta)
        print("h comp:", h_comp)
        print("grad_h_comp:", h_grad)
        print("h_dot:", h_dot)
        
        P = np.diag([1.0, 1.0, 1.0, 1000.0])
        q = -P @ np.concatenate((u_d, [0.0]))
        G = -np.concatenate((h_grad, [1]))
        h = alpha * h_comp + h_dot
        lb = 1.0 * np.array([-1,-1,-1, 0])
        ub = 1.0 * np.array([1, 1, 1, 10])
        print("Gu <", h)
        solution = solve_qp(P, q, G, h, ub=ub, lb=lb, solver="cvxopt")

        u = np.round(solution[:3], 2)
        slack = solution[3]
        print("==================================================================")

        return u, slack

        