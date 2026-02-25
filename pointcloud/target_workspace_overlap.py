import open3d as o3d
import numpy as np
import mujoco
from mujoco import viewer

pointcloud = np.load("./workspace_point_cloud_filtered.npy")
average = np.sum(pointcloud, 0)/pointcloud.shape[0]
print(average)

from scipy.spatial import ConvexHull
workspace_hull = ConvexHull(pointcloud)
A = workspace_hull.equations[:, :3]
b = workspace_hull.equations[:, 3]
print(A.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud)
workspace_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
o3d.visualization.draw_geometries([workspace_voxel_grid])

mesh = o3d.io.read_triangle_mesh("./target_meshes/target_sphere.stl")
target_voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.01)

workspace_voxels = workspace_voxel_grid.get_voxels()
target_voxels = target_voxel_grid.get_voxels()

# ===============================
model = mujoco.MjModel.from_xml_path("../unitree_rl_gym/resources/robots/g1_description/warehouse_scene.xml")
data = mujoco.MjData(model)

# viewer.launch(model, data)

data.qpos[3:7] = np.array([0.92388, 0, 0, 0.38268])
root_quat = data.qpos[3:7]

# get rotation matrix based on root orientation
R = np.eye(3)
# mujoco.mju_quat2Mat(R, root_quat)
# R = np.reshape(R, (3,3))
#viewer.launch(model, data)

voxel_size = 0.01

workspace_origin = workspace_voxel_grid.origin
workspace_coords = []
for voxel in workspace_voxels:
    v = voxel.grid_index
    pos = voxel_size * (v + 0.5) + workspace_origin
    pos_r = R @ pos
    workspace_coords.append(pos_r)

target_pos = [0.4, 0.2, 1.0]
target_origin = target_voxel_grid.origin + (target_pos - data.qpos[:3])
target_coords = []
for voxel in target_voxels:
    v = voxel.grid_index    
    pos = voxel_size * (v + 0.5) + target_origin
    target_coords.append(pos)

# voxel indexing based around canonical robot root frame
workspace_set = set(tuple(np.floor(coord/voxel_size).astype(int)) for coord in workspace_coords)
target_set = set(tuple(np.floor(coord/voxel_size).astype(int)) for coord in target_coords)

overlap = workspace_set & target_set
# print(voxel_size ** 3 * len(overlap))

workspace_pcd = o3d.geometry.PointCloud()
workspace_pcd.points = o3d.utility.Vector3dVector(workspace_coords)
workspace_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(workspace_pcd, voxel_size=0.01)
workspace_pcd.paint_uniform_color([0, 1, 0])  # green

target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(target_coords)
target_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(target_pcd, voxel_size=0.01)
target_pcd.paint_uniform_color([1, 0, 0])  # red

o3d.visualization.draw_geometries([workspace_pcd, target_pcd])

# ===========================================
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import open3d as o3d

pointcloud = np.load("./workspace_point_cloud_filtered.npy")
model = mujoco.MjModel.from_xml_path("../unitree_rl_gym/resources/robots/g1_description/warehouse_scene.xml")
data = mujoco.MjData(model)


#data.qpos[3:7] = np.array([0.92388, 0, 0, 0.38268])
root_quat = data.qpos[3:7]
# get rotation matrix based on root orientation
R = np.zeros(9)
mujoco.mju_quat2Mat(R, root_quat)
R = np.reshape(R, (3,3))

workspace_pos = (R @ pointcloud.T).T

target_pos = [0.4, 0.2, 1.0]
target_mesh = o3d.io.read_triangle_mesh("./target_meshes/target_sphere.stl")
target_points = np.unique(np.asarray(target_mesh.vertices), axis=0)  # Nx3 numpy array
target_points += (target_pos - data.qpos[:3])

workspace_hull = ConvexHull(workspace_pos)
A = workspace_hull.equations[:, :3]
b = workspace_hull.equations[:, 3]

# Extract hull vertices
hull_vertices = pointcloud[workspace_hull.vertices]

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    hull_vertices[:, 0],
    hull_vertices[:, 1],
    hull_vertices[:, 2]
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

inside_mask = np.all(A @ target_points.T + b[:, None] <= 1e-8, axis=0)
overlap_fraction = inside_mask.mean()
print(overlap_fraction)

# ==========================================================================================
pointcloud = np.load("./workspace_point_cloud_filtered.npy")

workspace_cloud = o3d.geometry.PointCloud()
workspace_cloud.points = o3d.utility.Vector3dVector(pointcloud)
workspace_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(workspace_cloud, 0.2)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
o3d.geometry.TriangleMesh.compute_vertex_normals(mesh)
o3d.io.write_triangle_mesh("./workspace_mesh.stl", mesh)

# target_pos = np.array([0.4329784,  0.02770339, 0.27684632])
# signed_distances = A @ target_pos + b
# print(signed_distances)

workspace_mesh = o3d.io.read_triangle_mesh("./workspace_mesh.stl")
workspace_mesh.compute_vertex_normals()
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(workspace_mesh))

# Query closest point
query_point = np.array([[1.0, 2.0, 0.5]])
query_point_t = o3d.core.Tensor(query_point, dtype=o3d.core.Dtype.Float32)

distance = scene.compute_distance(query_point_t).numpy()
print("Distance to mesh:", distance)  
closest_point = scene.compute_closest_points(query_point_t)['points'].numpy()
print(closest_point[0,:2])