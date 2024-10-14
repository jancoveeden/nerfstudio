# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.tensorf import TensoRFModel
from nerfstudio.models.depth_nerfacto import DepthNerfactoModel
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm
import glob

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

class Grid_Sampler:
    """
    Grid sampler to represent a scene as a grid of voxels. 
    Adapted from NeRAF.

    grid contains 4-channel 3D grid
        - Red ------> grid[0, :, : ,:]  --|
        - Green ----> grid[1, :, : ,:]  --|--|-- rgbsigma
        - Blue -----> grid[2, :, : ,:]  --|--|
        - Alpha ----> grid[3, :, : ,:]  --|

    Args:
        grid_limits: Scene box xyz limits as a list -> (x_min, x_max, y_min, y_max, z_min, z_max)
        max_res: int: Maximum resolution for x-, y-, z-axis
        device: Device where the grid must be stored
        nerf_name: Name of the NeRF model used
    """

    def __init__(self, grid_limits, max_res, device="cpu", nerf_name="nerfacto", dataset_type='hypersim') -> None:
        self.grid_limits =  grid_limits
        self.max_res = max_res
        self.device = device
        self.nerf_name = nerf_name

        self.scene_scale = 1.0
        self.bbox_scale = 1.0
        self.dataset_type = dataset_type

        max_pt = torch.Tensor(self.grid_limits[1::2])
        min_pt = torch.Tensor(self.grid_limits[::2])
        scale_factor = (self.max_res - 0.) / torch.max(max_pt - min_pt)
        max_xyz = ((max_pt - min_pt) * scale_factor).round().int()
        
        self.grid = torch.zeros(4, max_xyz[0], max_xyz[1], max_xyz[2], dtype=torch.float32, device=device)
        self.max_xyz = max_xyz

        # Visualization
        self.corners = None

        # Unused
        self.delta = 1e-2

    def density_to_alpha(self, density):
        """
        Converts queried density to alpha values

        Args:
            density: [batch_size, 1]
        """
        if (self.nerf_name == "tensorf"):
            # activation = np.clip(density, a_min=0, a_max=None)  # original NeRF uses RELU
            # alpha = np.clip(1.0 - torch.exp(-activation / 100.0), 0.0, 1.0)
            alpha = torch.clip(1.0 - torch.exp(-density * 1e-1), 0.0, 1.0)
        else:
            alpha = np.clip(1.0 - torch.exp(-torch.exp(density) / 100.0), 0.0, 1.0)  # NeRF-RPN instant-ngp
        return alpha
    
    def map_to_grid(self, xyz):
        """
        Maps the original range of xyz to the grid defined by max_xyz & grid_limits

        Args:
            xyz: Array to scale
        """
        xyz = (torch.Tensor(xyz) - self.grid_limits[::2]) / (self.grid_limits[1::2] - self.grid_limits[::2])
        grid_idxs = (xyz * (self.max_xyz - 1)).round().int()

        return grid_idxs

    def update_feature_grid(self, rays_xyz, rgb, density):
        """
        Converts queried rays to RGB and Alpha and adds them to an existing 3D feature grid

        Args:
            rays_xyz: Grid coordinates
            rgb: Average rgb values for given view directions
            density: Average density values for given view directions
        """
        rays_xyz = rays_xyz.reshape(-1, 3).to(self.device)  # [batch_size, 3]
        rgb = rgb.reshape(-1, 3).to(self.device)            # [batch_size, 3]
        density = density.reshape(-1, 1).to(self.device)    # [batch_size, 1]

        alpha = self.density_to_alpha(density)              # [batch_size, 1]  

        idxs = self.map_to_grid(rays_xyz)
        x_idxs, y_idxs, z_idxs = idxs[:, 0], idxs[:, 1], idxs[:, 2]

        self.grid[0, x_idxs, y_idxs, z_idxs] = rgb[:, 0].float().squeeze()    # [batch_size]
        self.grid[1, x_idxs, y_idxs, z_idxs] = rgb[:, 1].float().squeeze()    # [batch_size]
        self.grid[2, x_idxs, y_idxs, z_idxs] = rgb[:, 2].float().squeeze()    # [batch_size]
        self.grid[3, x_idxs, y_idxs, z_idxs] = alpha.float().squeeze()        # [batch_size]

    def get_obb_corners(self, center, extents, rot):
        corners = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ]) * extents 

        rotated_corners = (rot @ corners.T).T + center
        return rotated_corners
    
    def create_bbox_edges(self, base_idx):
        """
        Create edges (vertex pairs) for a bounding box.
        Returns list of vertex index pairs.
        """
        edges = []
        edges.extend([ # Bottom face
            (base_idx + 0, base_idx + 1),
            (base_idx + 1, base_idx + 2),
            (base_idx + 2, base_idx + 3),
            (base_idx + 3, base_idx + 0)
        ])
        edges.extend([ # Top face
            (base_idx + 4, base_idx + 5),
            (base_idx + 5, base_idx + 6),
            (base_idx + 6, base_idx + 7),
            (base_idx + 7, base_idx + 4)
        ])
        edges.extend([ # Vertical edges
            (base_idx + 0, base_idx + 4),
            (base_idx + 1, base_idx + 5),
            (base_idx + 2, base_idx + 6),
            (base_idx + 3, base_idx + 7)
        ])
        return edges

    def assign_points_to_grid(self, points, colors, opacities, show_boxes, crop, obbs=[], use_o3d=False):
        """
        Adds RGB and opacity values from corresponding pcd points to an existing 3D feature grid

        Args:
            points: Grid coordinates
            colors: RGB calculated for each point from SHS
            opacities: Each point's opacity
        """
        if use_o3d:
            self.grid_limits = crop
            CONSOLE.print(f"[bold blue]PLY bounds: {crop}")

        # Generate grid coordinates
        res = self.max_xyz.cpu().numpy()
        x = np.linspace(self.grid_limits[0], self.grid_limits[1], res[0])
        y = np.linspace(self.grid_limits[2], self.grid_limits[3], res[1])
        z = np.linspace(self.grid_limits[4], self.grid_limits[5], res[2])
        coords = np.meshgrid(x, y, z, indexing="ij")
        
        # Find the nearest grid point for each input point
        grid_points = np.vstack([coords[0].ravel(), coords[1].ravel(), coords[2].ravel()]).T
        tree = cKDTree(grid_points)
        _, indices = tree.query(points, k=1)

        # Assign colour and opacity to the grid
        for i, (color, opacity) in tqdm(enumerate(zip(colors, opacities)), total=len(colors),
                                        desc="Assigning points to grid", leave=False):
            # Find the 3D index of the closest grid cell
            idx = np.unravel_index(indices[i], res)
            self.grid[0:3, idx[0], idx[1], idx[2]] = torch.from_numpy(np.asarray(color))
            self.grid[3, idx[0], idx[1], idx[2]] = torch.from_numpy(np.asarray(opacity))

        if show_boxes and use_o3d:
            corners_grid_pts = []
            for i, obb in enumerate(obbs): 
                center = np.array(obb['center'])
                extents = np.array(obb['extent']) 
                rot = np.array(obb['rot'])

                world_corners = self.get_obb_corners(center, extents, rot)
                _, corner_idxs = tree.query(world_corners, k=1)
                corners_grid_pts.append([np.unravel_index(id, res) for id in corner_idxs])
            
            self.corners = corners_grid_pts

    def save_ply_with_bboxes(self, verts, colors, opacities, obbs, out_path, scene_name):
        """
        Saves 3DGS points as a .ply file with bounding boxes

        Args:
            verts: 3DGS xyz points
            colors: RGB values for each point
            opacities: Opacity values for each point
            bbox_dict: Dictionary containing bounding box information
            out_path: Folder to save the .ply file
            scene_name: Name of the scene
        """
        splat_path  = os.path.dirname(out_path)
        splat_path = os.path.join(splat_path, "splats_w_bboxes")
        os.makedirs(splat_path, exist_ok=True)
        splat_path = os.path.join(splat_path, f"{scene_name}.ply")

        bbox_vertices = []
        bbox_edges = []
        curr_vertex_count = len(verts)
        for idx, obb in enumerate(obbs):
            center = np.array(obb['center'])
            extents = np.array(obb['extent']) 
            rot = np.array(obb['rot'])

            box_verts = self.get_obb_corners(center, extents, rot)
            bbox_vertices.extend(box_verts)
            box_edges = self.create_bbox_edges(curr_vertex_count + idx * 8)
            bbox_edges.extend(box_edges)
        
        # Write .ply
        all_vertices = np.vstack([verts, bbox_vertices])

        bbox_colors = np.array([[1.0, 0.0, 0.0]] * len(bbox_vertices))  # Red color for bbox
        all_colors = (np.vstack([colors, bbox_colors]) * 255).astype(np.uint8)

        bbox_opacities = np.ones(len(bbox_vertices))                    # Fully opaque
        all_opacities = np.concatenate([opacities, bbox_opacities])

        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                        ('opacity', 'f4')]
        vertex_data = np.empty(len(all_vertices), dtype=vertex_dtype)
        vertex_data['x'] = all_vertices[:, 0]
        vertex_data['y'] = all_vertices[:, 1]
        vertex_data['z'] = all_vertices[:, 2]
        vertex_data['red'] = (all_colors[:, 0])
        vertex_data['green'] = (all_colors[:, 1])
        vertex_data['blue'] = (all_colors[:, 2])
        vertex_data['opacity'] = all_opacities

        edge_dtype = [('vertex1', 'i4'), ('vertex2', 'i4')]
        edge_data = np.array(bbox_edges, dtype=edge_dtype)

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        edge_element = PlyElement.describe(edge_data, 'edge')

        PlyData([vertex_element, edge_element], text=True).write(splat_path)
        CONSOLE.print(f"[bold green]:white_check_mark: Saved splat with boxes at {splat_path}")

    def generate_coords(self):
        """
        Generates xyz coordinates for a point grid
        """
        x = torch.linspace(self.grid_limits[0], self.grid_limits[1], self.max_xyz[0], dtype=torch.float32)
        y = torch.linspace(self.grid_limits[2], self.grid_limits[3], self.max_xyz[1], dtype=torch.float32)
        z = torch.linspace(self.grid_limits[4], self.grid_limits[5], self.max_xyz[2], dtype=torch.float32)
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        grid_coords = torch.stack([x, y, z], dim=-1).reshape(3, self.max_xyz[0],self.max_xyz[1],self.max_xyz[2]).cpu()

        return grid_coords

    def get_nerf_rpn_output(self):
        """
        Returns rgbsigma in a format similar to NeRF-RPN
        Converts from (res_x, res_y, res_z, 4) -> (res_x * res_y * res_z, 4)
        Refer to https://github.com/lyclyc52/NeRF_RPN
        """
        # Returns (W * L * H, 4) = (res_x * res_y * res_z, 4)
        rgbsigma = self.grid[:4,:,:,:]
        rgbsigma = rgbsigma.view(-1, 4).cpu().numpy()

        return rgbsigma

    def get_plotly_box_corners(self, obb, out_type='obb'):
        """
        Compute 8 corners and 12 edges
        
        position: center of obb
        extents: size of box in each direction from its center to outer surface xyz
        orientation: rotation matrix
        """
        position = np.array(obb['center'])
        extents = np.array(obb['extent'])
        orientation = np.array(obb['rot'])

        corners = np.array([
                        [-extents[0], -extents[1], -extents[2]],
                        [extents[0], -extents[1], -extents[2]],
                        [extents[0], extents[1], -extents[2]],
                        [-extents[0], extents[1], -extents[2]],
                        [-extents[0], -extents[1], extents[2]],
                        [extents[0], -extents[1], extents[2]],
                        [extents[0], extents[1], extents[2]],
                        [-extents[0], extents[1], extents[2]]
                    ])
        
        if (out_type == 'aabb'):
            corners = corners + position
        else:
            corners = np.dot(corners, orientation.T) + position

        if self.dataset_type == 'hypersim':
            corners = (corners - self.grid_limits[::2]) / (self.grid_limits[1::2] - self.grid_limits[::2])

        corners = np.array( (torch.Tensor(corners) * (self.max_xyz - 1)) )

        edge_planes = [[0, 1], [1, 2], [2, 3], [3, 0],  # bottom edges
                       [4, 5], [5, 6], [6, 7], [7, 4],  # top edges
                       [0, 4], [1, 5], [2, 6], [3, 7]]  # connecting edges

        return corners, edge_planes

    def plot_point_cloud(self, alpha_threshold=0.5, show_boxes=False, obbs=[]):
        """
        Visualizes the extracted feature grid as point cloud

        Args:
            alpha_threshold: Density threshold at which points are saved
            show_boxes: Whether to plot the scene bounding box and coordinate frame
        """
        import open3d as o3d
        grid = self.grid.cpu().detach().numpy()

        W, L, H = grid.shape[1:]
        x, y, z = np.mgrid[0:W, 0:L, 0:H]

        mask = (grid[3].flatten() > alpha_threshold) 
        x, y, z = x.flatten()[mask], y.flatten()[mask], z.flatten()[mask]

        rgb = grid[:3, :, :, :].reshape(3, -1).T 
        rgb = rgb[mask]

        pcd = o3d.geometry.PointCloud()
        points = np.vstack((x, y, z)).T
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis_list = [pcd]

        if (show_boxes):
            # x-axis: red, y-axis: green, z-axis: blue
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
            vis_list.append(coordinate_frame)

            # Point cloud scenebox
            min_pt = np.array([0, 0, 0])
            max_pt = np.array([W, L, H])
            scene_box = o3d.geometry.AxisAlignedBoundingBox(min_pt, max_pt)
            scene_box.color = (0, 1, 0) # green
            vis_list.append(scene_box)

            # Object bounding boxes
            # if (self.dataset_type == 'hypersim'):
            #     for obb in obbs:
            #         pos = np.array(obb['center']) * max_pt
            #         extents = np.array(obb['extent']) * max_pt
            #         rot = np.array(obb['rot'])
            #         obb_box = o3d.geometry.OrientedBoundingBox(center=pos, R=rot, extent=extents)
            #         obb_box.color = (1, 0, 0)  # red
            #         vis_list.append(obb_box)

            if (self.corners is not None):
                for c in self.corners:
                    corn_np = np.array(c).astype(np.float64)
                    if len(corn_np) < 8:
                        print("Warning: Not enough corners to create an OrientedBoundingBox. Skipping box.")
                        continue
                    if np.unique(corn_np, axis=0).shape[0] < 8:
                        print("Warning: Corners are not unique enough to form a valid box. Skipping box.")
                        continue

                    try:
                        pts_ = o3d.utility.Vector3dVector(np.array(c))
                        obb = o3d.geometry.OrientedBoundingBox.create_from_points(pts_)
                        obb.color = [0, 0, 1]  # Green 
                        vis_list.append(obb)
                    except Exception as e:
                        print(f"Error creating OrientedBoundingBox: {e}")

        o3d.visualization.draw_geometries(vis_list)

    def visualize_depth_grid(self, db_outs, obbs, show_poses=False, show_boxes=False, box_type='obb'):
        """
        Visualizes the extracted feature grid as depth grid

        Args:
            db_outs: Nerfstudio pipeline.datamanager.train_dataparser_outputs object
            obbs: List of dictionaries of oriented bboxes for each object
            show_poses: Whether to plot the camera poses in the grid
            show_boxes: Whether to plot the object bounding boxes in the grid
            box_type: Type of bounding box to plot (aabb or obb)
        """
        import plotly.graph_objects as go

        grid = self.grid.cpu().detach().numpy()
        x, y, z = np.mgrid[0:grid.shape[1], 0:grid.shape[2], 0:grid.shape[3]]
        
        # Scene Data
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=grid[3].flatten(), 
            opacity=0.1,
            surface_count=17,
            colorscale='Viridis'
        ))

        # Add camera positions as dots
        if (show_poses):
            camera_positions = []
            for cam in db_outs.cameras:
                pose = cam.camera_to_worlds.cpu().numpy()
                camera_pos = pose[:3, 3] / 3.
                camera_positions.append(camera_pos)
            camera_positions = torch.Tensor(np.array(camera_positions))
            cam_poses = (camera_positions - self.grid_limits[::2]) / (self.grid_limits[1::2] - self.grid_limits[::2])
            cam_poses = (cam_poses * (self.max_xyz - 1))

            fig.add_trace(go.Scatter3d(
                x=cam_poses[:, 0],
                y=cam_poses[:, 1],
                z=cam_poses[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    symbol='circle'
                ),
                name='Camera',
                showlegend=False
            ))
        
        # Add object bounding boxes
        if (show_boxes):
            for i, obb in enumerate(obbs):
                # pos = np.array(obb.T) * bbox_scale * scene_scale
                # if ((pos < self.grid_limits[::2]).any() or (pos > self.grid_limits[1::2]).any()):
                #     continue
                vertices, edges = self.get_plotly_box_corners(obb, box_type)

                for edge in edges:
                    fig.add_trace(go.Scatter3d(
                        x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                        y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                        z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ))
                
                fig.add_trace(go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='blue'),
                    showlegend=False
                ))

        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""

def get_ngp_obj_bbox(xform, extent):
    """
    Get AABB from the OBB of an object.
    
    Args:
        xform: 3x4 matrix for rotation, translation
        extent: 1x3 matrix for length, width, height (xyz) of bbox
    
    Returns:
        min_pt: 1x3 matrix
        max_pt: 1x3 matrix
    """
    corners = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1],
    ], dtype=float).T # 3x8

    corners *= np.expand_dims(extent, 1) * 0.5 #3x8 * 3x1 = 3x8
    corners = xform[:, :3] @ corners + xform[:, 3, None] #3x3 * 3x8 * 3x1 = 3x8

    return np.min(corners, axis=1), np.max(corners, axis=1)

def estimate_scene_box(json_dict, nerf_name, dataset_type, db_outs, margin=0.1):
    """
    Estimates scene bounding box using 
        - Object bounding boxes
        - Cameras
    
    Returns:
        min_pt: 1x3 matrix
        max_pt: 1x3 matrix
    """
    scene_scale_f = db_outs.dataparser_scale
    transform_matrix = db_outs.dataparser_transform
    scales = {"nerfacto": 1.0, "depth-nerfacto": 1.0, "tensorf": 1.0, 
              "zipnerf": 2.0, "splatfacto": 2.0, "pynerf": 1.0}

    if (dataset_type == 'scannet'):
        mini = [ins['min_pt'] for ins in json_dict['instances']]
        maxi = [ins['max_pt'] for ins in json_dict['instances']]

        min_pts = torch.Tensor(mini)
        max_pts = torch.Tensor(maxi)

        min_pt = min_pts.min(dim=0)[0]
        max_pt = max_pts.max(dim=0)[0]
        #CONSOLE.print(f"[bold magenta]Estimated scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")

        if nerf_name in ['splatfacto', 'zipnerf']:
            scales["splatfacto"] = 10.0
            scales["zipnerf"] = 10.0
            corners = torch.tensor([
                [min_pt[0], min_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]]
            ])
            corners_homog = torch.cat((corners, torch.ones_like(corners[:, :1])), dim=1)
            transf_corners = (corners_homog @ transform_matrix.T) * scene_scale_f
            min_pt = torch.min(transf_corners[:, :3], dim=0)[0]
            max_pt = torch.max(transf_corners[:, :3], dim=0)[0]
            # min_pt_ = torch.cat((min_pt, torch.tensor([1.0])))
            # min_pt = (min_pt_ @ transform_matrix.T) * scene_scale_f
            # max_pt_ = torch.cat((max_pt, torch.tensor([1.0])))
            # max_pt = (max_pt_ @ transform_matrix.T) * scene_scale_f
        else:
            min_pt = (min_pt - max_pt) / 10.0
            max_pt = ((max_pt + (scene_scale_f * max_pt)) / 10.0) + min_pt

    elif (dataset_type == 'hypersim'):
        min_box_pt = []
        max_box_pt = []
        min_cams_pt = []
        max_cams_pt = []

        # Get min & max corners from bbox:
        for obj in json_dict['bounding_boxes']:
            extent = np.array(obj['extents'])
            rot = np.array(obj['orientation'])
            position = np.array(obj['position'])

            xform = np.hstack([rot, np.expand_dims(position, 1)])
            min_pt_, max_pt_ = get_ngp_obj_bbox(xform, extent)
            min_box_pt.append(min_pt_)
            max_box_pt.append(max_pt_)
        min_box_pt = np.min(np.array(min_box_pt) , axis=0) # 1x3
        max_box_pt = np.max(np.array(max_box_pt), axis=0) # 1x3

        # Get min & max corners from cameras:
        camera_pos = []
        for frame in json_dict['frames']:
            xform = np.array(frame['transform_matrix'])
            camera_pos.append(xform[:3, 3])
        camera_pos = np.array(camera_pos)
        min_cams_pt = np.min(camera_pos, axis=0) # 1x3
        max_cams_pt = np.max(camera_pos, axis=0) # 1x3

        # Save the minimum and maximum corners
        min_pt = torch.from_numpy(np.minimum(min_box_pt, min_cams_pt))
        max_pt = torch.from_numpy(np.maximum(max_box_pt, max_cams_pt))
        #CONSOLE.print(f"[bold magenta]Estimated scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")
        
        # Enlarge + re-center scenebox
        max_diff = torch.max(max_pt - min_pt)
        scale_f = (scales.get(nerf_name, 1.0)+1.0) / max_diff
        scale_f = scale_f + (margin+0.1 / max_diff)
        z_scale = scale_f #+ (margin+0.2 / max_diff)
        center = (min_pt + max_pt) / 2.0

        for i in range(3):
            sign = 1 if center[i] <= 0 else -1
            scale = z_scale if i == 2 else scale_f
            #scale = scale_f
            min_pt[i] = (min_pt[i] + (sign * center[i])) * scale
            max_pt[i] = (max_pt[i] - (sign * center[i])) * scale
        
        # Check if z-axis is too small
        diff_z = max_pt[2] - min_pt[2]
        if (diff_z <= 0.3):
            min_pt[2] -= (0.3 - diff_z)/2.0
            max_pt[2] += (0.3 - diff_z)/2.0
    
    # Clamp to [-1, +1] for all nerf models except for zipnerf [-2, +2]
    min_pt = torch.clamp(min_pt, min=-scales.get(nerf_name, 1.0), max=scales.get(nerf_name, 1.0))
    max_pt = torch.clamp(max_pt, min=-scales.get(nerf_name, 1.0), max=scales.get(nerf_name, 1.0))
    #CONSOLE.print(f"[bold magenta]Estimated scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")

    return min_pt, max_pt 

def generate_fixed_viewdirs():
    """
    Generates 18 fixed view directions for RGBSigma sampling
    """
    phis = [np.pi / 3, 0, -np.pi]
    thetas = [k * np.pi / 3 for k in range(0, 6)]
    viewdirs = []

    for phi in phis:
        for theta in thetas:
            viewdirs.append(torch.Tensor([
                np.cos(phi) * np.sin(theta),
                np.cos(phi) * np.sin(theta),
                np.sin(theta)
            ]))
    dirs = torch.stack(viewdirs, dim=0)
    return dirs

def generate_pcd_viewdirs(n_points=5):
    """
    Generates 18 fixed view directions for point cloud cameras
    """
    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(0, n_points, dtype=float) + 0.5
    
    phi = np.arccos(1 - 2 * i / n_points)
    theta = 2 * np.pi * i / golden_ratio
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    viewdirs = np.column_stack((x, y, z))
    return torch.from_numpy(viewdirs).float()

def parse_scannet_pcd_objects(base_path, scene_id, max_xyz, scene_scale):
    """
    Reads ScanNet scene objects, labels, ids from the point cloud 
    and returns object labels, centers, extents

    Args:
        base_path: Path to the specific folder containing all ScanNet scenes
        scene_id: Name of specific scene
        max_xyz: Maximum xyz values for the scene
        scene_scale: Scale of the scene from dataparser
    """
    import open3d as o3d
    max_xyz = max_xyz.cpu().numpy()

    base_path = os.path.join(base_path, scene_id)
    pcd_path_ = os.path.join(base_path, f"{scene_id}_vh_clean_2.ply")
    pcd = o3d.io.read_point_cloud(pcd_path_)
    points = np.asarray(pcd.points)
    ply_scene_box = pcd.get_axis_aligned_bounding_box()
    with open(os.path.join(base_path, f"{scene_id}_vh_clean_2.0.010000.segs.json"), "r") as f:
        segs_data = json.load(f)
    with open(os.path.join(base_path, f"{scene_id}.aggregation.json"), "r") as f:
        agg_data = json.load(f)
    
    pcd_min = ply_scene_box.min_bound
    pcd_max = ply_scene_box.max_bound

    objects = []
    for obj in agg_data['segGroups']:
        # Calculate object AABB
        obj_points = points[np.isin(segs_data['segIndices'], obj['segments'])]
        min_bound = np.min(obj_points, axis=0)
        max_bound = np.max(obj_points, axis=0)
        
        # Calculate center and extents
        center = (min_bound + max_bound) / 2
        extent = max_bound - min_bound
        
        # Scale to grid size
        scaled_center = np.array(center / np.max(points, axis=0))
        scaled_extent = np.array(extent / np.max(points, axis=0))

        obb = np.concatenate((scaled_center, scaled_extent), axis=0)
        objects.append(obb)
        
    return objects, pcd_min, pcd_max

def parse_transforms_to_obbs(bbox_dict, max_xyz, dataset_type='hypersim', min_extent=0.2, bbox_scale=0.5, 
                             scene_scale=0.3333, scene_name='ai_001_001', use_pcd_objects=False,
                             vis_method='pcd'):
    """
    Converts the extents, orientation and position in the 
    transform.json/obj_instances.json to oriented bounding boxes (OBBs)

    Args:
        bbox_dict: json_dict
        max_xyz: Maximum grid resolution for a scene
        dataset_type: Dataset type (hypersim, scannet)
        min_extent: Minimum extent for filtering out small objects
        bbox_scale: Scale factor for bounding boxes
        scene_scale: Scale factor for the scene
        scene_name: Name of the scene
        use_pcd_objects: Whether to use pcd objects metadata for scannet scene
        vis_method: Visualization method (plotly, pcd)
    """
    obbs = []

    if (dataset_type == 'hypersim'):
        bboxes = bbox_dict["bounding_boxes"]
        for obj in bboxes:
            orientation = np.array(obj['orientation'])
            extents = np.array(obj['extents']) * scene_scale * bbox_scale
            position = np.array(obj['position']) * scene_scale * bbox_scale

            if (extents < min_extent).any(): # Filters out small objects
                continue

            obb = {
                'center': position,
                'extent': extents,
                'rot': orientation
            }
            obbs.append(obb)

    elif (dataset_type == 'scannet'):
        instances = bbox_dict['instances']
        obb = np.array([x['obb'] for x in instances])

        if (vis_method == 'plotly'):
            # Normalize the OBBs
            max_pt = np.array([x['max_pt'] for x in instances])
            min_pt = np.array([x['min_pt'] for x in instances])
            bbox_min = np.min(min_pt, axis=0)
            bbox_max = np.max(max_pt, axis=0) 
            aabb_lengths = bbox_max - bbox_min 
            obb[:, :3] = (obb[:, :3] - bbox_min) / aabb_lengths 
            obb[:, 3:6] = (obb[:, 3:6]) / aabb_lengths

            labels = [x['label'] for x in instances]
            inc_labels = ['bed', 'chair', 'table']

        # Parse pcd objects
        if use_pcd_objects:
            base_pcd_path = r"E:\NeRF_datasets\mmdetection3d\data\scannet\scans"
            obbs, pcd_min, pcd_max = parse_scannet_pcd_objects(base_pcd_path, scene_name, 
                                                              max_xyz, scene_scale)
        
        for i, ob in enumerate(obb):  
            center = ob[:3] 
            extents = ob[3:6]

            angle = ob[6]
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,              0,             1]
            ])

            # if np.min(extents) < min_extent: # Filters out small objects
            #     continue
            if (vis_method == 'plotly'):
                if not labels[i] in inc_labels:
                    continue
                print(f"label:{labels[i]}\ncenter:{center}\nextents:{extents}\n")
            
            obb = {
                'center': center,
                'extent': extents,
                'rot': rot
            }
            obbs.append(obb)
    
    if len(obbs) == 0:
        CONSOLE.print(f"[bold red]No obbs found in {scene_name}. \
                        Try decreasing the --min_vis_bbox.")

    return obbs

def extract_3d_gaussians(model, sampler, coords, cams, visualize_gausses=False):
    """
    Can be used to visualize 3D gaussians in plotly
    """
    
    from sklearn.neighbors import NearestNeighbors

    with torch.no_grad():    
        positions = model.means.cpu().numpy()
        colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy() 
        opacities = model.opacities.data.cpu().numpy()

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(positions)
        dists_, indices_ = nbrs.kneighbors(coords)

        rgb = torch.from_numpy(colors[indices_.flatten()])
        density = torch.from_numpy(opacities[indices_.flatten()])
        sampler.update_feature_grid(coords, rgb, density)

    return sampler

def shs_3_to_rgb(f_dc_0, f_dc_1, f_dc_2):
    SH_C0 = 0.28209479177387814
    r = 0.5 + SH_C0 * f_dc_0
    g = 0.5 + SH_C0 * f_dc_1
    b = 0.5 + SH_C0 * f_dc_2
    return r, g, b

def compute_pixel_area(dir, nerf_model):
    pixel_area = None
    if (nerf_model == 'pynerf'):
        # Simulating neighbouring pixel ray directions
        delta = 1e-5
        dx = torch.sqrt(torch.sum((dir + delta - dir) ** 2, dim=-1))
        dy = torch.sqrt(torch.sum((dir + delta - dir) ** 2, dim=-1))
        pixel_area = (dx * dy)[..., None] # ("num_rays":..., 1)
    
    return pixel_area

def query_nerf_model(nerf_name, model, pipeline, json_path, output_path, config_path,
                     dataset_type='hypersim', max_res=128, batch_size=4096, min_bbox=0.2, crop_scene=True, 
                     use_fixed_viewdirs=False, visualize=False, show_poses=False, show_boxes=False, 
                     vis_method="plotly", box_type='obb', ply_color_mode='sh_coeffs', use_splat_o3d=True,
                     save_splat_boxes=True):
    """
    Extracts rgbsigma from a given pre-trained NeRF scene

    Args:
        nerf_name: Name of NeRF model used for novel view synthesis
        model: NeRF model object
        pipeline: NeRF model pipeline module
        json_path: Path to bbox metadata json containing bboxes, transformation matrices etc.
        output_path: Path to .npz file to save
        dataset_type: Dataset name: ['hypersim', 'scannet']
        max_res: Maximum resolution for an axis for the feature grid 
        batch_size: Batch size of grid coordinates used for feature grid extraction
        min_bbox: Minimum length/extent of an object bounding box to visualize
        crop_scene: Whether to crop the scene bounding box or not.
        use_fixed_viewdirs: Whether to generate & use fixed view directions
        visualize: Whether to plot the 3D feature grid or just save the rgbsigma in .npz file
        show_poses: Whether to plot camera positions in plotly visualization
        show_boxes: Whether to plot object bboxes in plotly visualization
        vis_method: Which method to use to plot the 3D feature grid
        box_type: Whether to visualize obb or aabb
        ply_color_mode: Color mode for the point cloud visualization
        save_ply_o3d: Whether to save the point cloud in .ply format using Open3D
    """
    device = model.device
    db_outs = pipeline.datamanager.train_dataparser_outputs
    dataparser_scale = db_outs.dataparser_scale
    cams = db_outs.cameras
    scene_name = os.path.basename(output_path)[:-4]
    pcd_thresh = 0.5
    num_pts = 0
    obbs = []
    trf = nerf_name == "tensorf"
    zp = nerf_name == "zipnerf"

    with open(json_path) as f:
        json_dict = json.load(f) 
    
    # Get estimated nerfstudio scene bbox or predefined scene bbox
    if (crop_scene):
        min_pt, max_pt = estimate_scene_box(json_dict, nerf_name, dataset_type, db_outs)
    else:
        min_pt = db_outs.scene_box.aabb[0] #- 1.0
        max_pt = db_outs.scene_box.aabb[1] #+ 1.0
    aabb_lengths = max_pt - min_pt
    grid_limits = np.array([min_pt[0],max_pt[0],min_pt[1],max_pt[1],min_pt[2],max_pt[2]])

    # Generate [4, res_x, res_y, res_z] grid and xyz coordinates
    if (nerf_name == "splatfacto"):
        gaus_model = ExportGaussianSplat(config_path, os.path.dirname(output_path), ply_color_mode=ply_color_mode)
        verts, pts, colors, opacities, crop = gaus_model.extract(model, scene_name, grid_limits, 
                                                                 db_outs, crop_scene, use_splat_o3d)

    # Generate [4, res_x, res_y, res_z] grid and xyz coordinates
    grid_sampler = Grid_Sampler(grid_limits, max_res, "cpu", nerf_name, dataset_type) 
    grid_coords = grid_sampler.generate_coords()
    coords_to_render = grid_coords.view(-1, 3) # [res_x*res_y*res_z, 3] 

    # Parse bboxes and get scene + bbox scale
    if dataset_type == 'hypersim':
        # Reverse of: 0.5 * meta.get("aabb_scale", 1)
        bbox_scale = 2.0 * (1/float(json_dict["aabb_scale"]))
        bbox_scale *= 2.0 if zp else 1.0
    elif dataset_type == 'scannet':
        bbox_scale = 0.5
    
    if (show_boxes) and (visualize):
        grid_sampler.scene_scale = dataparser_scale
        grid_sampler.bbox_scale = bbox_scale
        obbs = parse_transforms_to_obbs(json_dict, grid_sampler.max_xyz, dataset_type, min_bbox, 
                                        bbox_scale, dataparser_scale, scene_name, False, vis_method)
    
    # Enables SceneBox positions for Nerfacto
    if (nerf_name in ['nerfacto', 'depth-nerfacto', 'pynerf']):
        model.field.spatial_distortion = None 

    # Crop scene boundaries
    if (nerf_name in ['nerfacto', 'tensorf', 'depth-nerfacto']):
        min_pt_ = torch.Tensor([-1., -1., -1.])
        max_pt_ = torch.Tensor([1., 1., 1.])
        aabb_lengths = max_pt_ - min_pt_

    # Get camera view directions
    camera_views = [] 
    if (use_fixed_viewdirs):
        camera_views = generate_fixed_viewdirs() # [18, 3]
    else:
        for j in range(cams.size):
            trans_mat = cams[j].camera_to_worlds
            viewdir = torch.Tensor(torch.Tensor(trans_mat[:3, :3]) @ torch.Tensor([0, 0, -1])) # [3]
            camera_views.append(viewdir) # [num_train_views, 3]

    # if (dataset_type == 'scannet'):
    #     camera_views = camera_views[1::2]
    #CONSOLE.print(f"[bold blue]Number of dataset images: {len(db_outs.image_filenames)}")

    CONSOLE.print(f"[bold blue]Batch size set to: {batch_size}")
    CONSOLE.print(f"[bold blue]Number of views to extract from: {len(camera_views)}")
    CONSOLE.print(f"[bold magenta]Using scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")

    if (nerf_name == "splatfacto"):
        pcd_thresh = 0.2
        grid_sampler.assign_points_to_grid(pts, colors, opacities, show_boxes,
                                           crop, obbs, use_splat_o3d)
        if use_splat_o3d and save_splat_boxes:
            grid_sampler.save_ply_with_bboxes(verts, colors, opacities, obbs, 
                                              os.path.dirname(output_path), scene_name)
        # To visualize 3D gaussians:
        # grid_sampler = extract_3d_gaussians(model, grid_sampler, coords_to_render, cams)
    else:
        # Extract RGB and density from NeRF into the 3D feature grid
        with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),TimeElapsedColumn(),MofNCompleteColumn(),
                    transient=True,
        ) as progress:
            task = progress.add_task("[green]Extracting feature grid...", total=coords_to_render.shape[0])
            for i in range(0, coords_to_render.shape[0], batch_size):
                # Check for the last batch
                if i + batch_size > coords_to_render.shape[0]:
                    batch_size = coords_to_render.shape[0] - i

                # Construct ray batch
                batch_coords = coords_to_render[i:i+batch_size]                     # [batch_size, 3]
                rgbs = []
                densities = []  
                if zp:
                    batch_coords = batch_coords.unsqueeze(0).to(device)             # [1, batch_size, 3]
                    batch_coords = torch.permute(batch_coords, (1, 0, 2))           # [batch_size, 1, 3]
                    batch_std = torch.full_like(batch_coords[..., 0], 0.0)[:, None] # [batch_size, 1, 1]
                    batch_std = batch_std.to(device)
                else:
                    ori = (batch_coords *  aabb_lengths)            # [batch_size, 3]
                    start = torch.zeros_like(ori)                   # [batch_size, 3]
                    end = torch.zeros_like(ori)                     # [batch_size, 3]

                # Iterate through camera views
                for v in camera_views:
                    viewdir = v.to(model.device) # [3]
                    
                    with torch.no_grad():
                        if zp:
                            field_outs = model.zipnerf.nerf_mlp(rand=False, means=batch_coords,  
                                                                stds=batch_std, viewdirs=viewdir)
                            rgb = field_outs['rgb']
                            density = field_outs['density'].unsqueeze(1)
                        else:
                            dir = viewdir.expand(batch_size, -1)  # [3] -> [batch_size, 3]
                            pixel_area = compute_pixel_area(dir, nerf_name)
                            f = Frustums(ori, dir, start, end, pixel_area)
                            cam_indices = torch.zeros((batch_size, 1), dtype=torch.int)
                            rays = RaySamples(f, cam_indices).to(model.device)
                            field_outs = model.field.forward(rays, nerf_rgbd=True) if trf else model.field.forward(rays)
                            rgb = field_outs[FieldHeadNames.RGB]
                            density = field_outs[FieldHeadNames.DENSITY]

                        density = density.unsqueeze(1) if trf else density

                        rgbs.append(rgb.cpu())
                        densities.append(density.cpu())

                # Add features to grid_sampler
                rgb = torch.mean(torch.stack(rgbs, dim=0), dim=0)            # [batch_size, 3]
                density = torch.mean(torch.stack(densities, dim=0), dim=0)   # [batch_size, 1]
                grid_sampler.update_feature_grid(batch_coords, rgb, density) 
                progress.update(task, advance=batch_size)
                torch.cuda.empty_cache()

    if (visualize):
        # Visualize the extracted 3D feature grid
        CONSOLE.print(f"[bold green]:white_check_mark: Visualizing grid with size: {grid_sampler.max_xyz} and model: {nerf_name}")
        if (vis_method == 'plotly'):
            grid_sampler.visualize_depth_grid(db_outs, obbs, show_poses, show_boxes, box_type)
        else:
            grid_sampler.plot_point_cloud(pcd_thresh, show_boxes, obbs)
    else:
        # Save rgbsigma
        CONSOLE.print(f"[bold green]Successfully extracted rgbsigma from {nerf_name}")
        rgbsigma = grid_sampler.get_nerf_rpn_output()
        min_pt = min_pt.cpu().numpy()
        max_pt = max_pt.cpu().numpy()
        res = grid_sampler.max_xyz.cpu().tolist()
        scale = grid_sampler.scene_scale
        CONSOLE.print(f"[bold green]Feature grid resolution: {grid_sampler.grid.size()}")
        CONSOLE.print(f"[bold green]Saving rgbsigma with size: {rgbsigma.shape}")
        np.savez_compressed(output_path, rgbsigma=rgbsigma, resolution=res,
                            bbox_min=min_pt, bbox_max=max_pt,
                            scale=scale, offset=0.0)
        CONSOLE.print(f"[bold green]:white_check_mark: Saved features for scene: {scene_name}")
        

def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        metadata = {"directions_norm": torch.linalg.vector_norm(directions, dim=-1, keepdim=True)}
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        import open3d as o3d

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        import open3d as o3d
        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        filename = self.output_dir / "splat.ply"

        map_to_tensors = OrderedDict()

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if self.ply_color_mode == "rgb":
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                colors = (colors * 255).astype(np.uint8)
                map_to_tensors["red"] = colors[:, 0]
                map_to_tensors["green"] = colors[:, 1]
                map_to_tensors["blue"] = colors[:, 2]
            elif self.ply_color_mode == "sh_coeffs":
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            if model.config.sh_degree > 0:
                if self.ply_color_mode == "rgb":
                    CONSOLE.print("Warning: model has higher level of spherical harmonics. Using RGB colors")
                elif self.ply_color_mode == "sh_coeffs":
                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]

                n = map_to_tensors["x"].shape[0]
                count = n

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
        nan_count = np.sum(select) - n

        # filter gaussians that have opacities < 1/255, because they are skipped in cuda rasterization
        low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < -5.5373  # logit(1/255)
        lowopa_count = np.sum(low_opacity_gaussians)
        select[low_opacity_gaussians] = 0

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)
    
    def extract(self, model, scene_name, crop_bounds, db_outs, crop_scene=True, use_o3d=True, overwrite=True):
        """
        This function does the following:
            - Exports splat.ply file from the model.
            - Converts spherical harmonics to RGB colors.

        Args:
            model: Splatfacto model
            scene_name: Dataset scene_name
            crop_bounds: 3D feature grid cropping bounds
        """
        splat_path  = os.path.dirname(self.output_dir)
        splat_path = os.path.join(splat_path, "splats")
        splat_path = os.path.join(splat_path, f"{scene_name}.ply")

        if os.path.exists(splat_path) == False or overwrite == True:
            count = 0
            map_to_tensors = OrderedDict()

            with torch.no_grad():
                positions = model.means.cpu().numpy()
                count = positions.shape[0]
                n = count

                if use_o3d:
                    map_to_tensors["positions"] = positions
                    map_to_tensors["normals"] = np.zeros_like(positions, dtype=np.float32)
                else:
                    map_to_tensors["x"] = positions[:, 0]
                    map_to_tensors["y"] = positions[:, 1]
                    map_to_tensors["z"] = positions[:, 2]
                    map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
                    map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
                    map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)
                map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

                scales = model.scales.data.cpu().numpy()
                for i in range(3):
                    map_to_tensors[f"scale_{i}"] = scales[:, i, None]

                quats = model.quats.data.cpu().numpy()
                for i in range(4):
                    map_to_tensors[f"rot_{i}"] = quats[:, i, None]
                
                if self.ply_color_mode == "rgb":
                    colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                    colors = (colors * 255).astype(np.uint8)
                    if not use_o3d:
                        map_to_tensors["red"] = colors[:, 0]
                        map_to_tensors["green"] = colors[:, 1]
                        map_to_tensors["blue"] = colors[:, 2]
                    else:
                        map_to_tensors["colors"] = colors
                elif self.ply_color_mode == "sh_coeffs":
                    shs_0 = model.shs_0.contiguous().cpu().numpy()
                    for i in range(shs_0.shape[1]):
                        map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                if model.config.sh_degree > 0:
                    if self.ply_color_mode == "rgb":
                        CONSOLE.print("Note: Splatfacto model contains higher level of spherical harmonics")
                    elif self.ply_color_mode == "sh_coeffs":
                        # transpose(1, 2) was needed to match the sh order in Inria version
                        shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                        shs_rest = shs_rest.reshape((n, -1))
                        for i in range(shs_rest.shape[-1]):
                            map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

                if crop_scene:
                    min_corner = crop_bounds[::2]
                    max_corner = crop_bounds[1::2]
                    center = tuple((min_corner + max_corner) / 2.0)
                    scale = tuple(max_corner - min_corner)
                    rotation = (0.0, 0.0, 0.0)
                    if center is not None and rotation is not None and scale is not None:
                        crop_obb = OrientedBox.from_params(center, rotation, scale)
                        assert crop_obb is not None
                        mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                        for k, t in map_to_tensors.items():
                            map_to_tensors[k] = map_to_tensors[k][mask]
                        n = map_to_tensors["positions"].shape[0] if use_o3d else map_to_tensors["x"].shape[0]
                        count = n

            # Remove NaN/Inf values & low opacity gaussians
            select = np.ones(n, dtype=bool)
            for k, t in map_to_tensors.items():
                n_before = np.sum(select)
                select = np.logical_and(select, np.isfinite(t).all(axis=-1))
                n_after = np.sum(select)
                if n_after < n_before:
                    CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")
            low_opacity_gaussians = (map_to_tensors["opacity"]).squeeze(axis=-1) < -5.5373
            select[low_opacity_gaussians] = 0

            if np.sum(select) < n:
                CONSOLE.print(f"Values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][select]
                count = np.sum(select)

            # Check for correct number of elements
            if not all(len(tensor) == count for tensor in map_to_tensors.values()):
                raise ValueError("Count does not match the length of all tensors")

            if not all(isinstance(tensor, np.ndarray) and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
                and tensor.size > 0 for tensor in map_to_tensors.values()
            ):
                raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

            # Write .ply file
            if not use_o3d:
                with open(splat_path, "wb") as ply_file:
                    ply_file.write(b"ply\n")
                    ply_file.write(b"format binary_little_endian 1.0\n")
                    ply_file.write(f"element vertex {count}\n".encode())

                    for key, tensor in map_to_tensors.items():
                        data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                        ply_file.write(f"property {data_type} {key}\n".encode())
                    ply_file.write(b"end_header\n")

                    for i in tqdm(range(count), total=count, desc="Writing points to splat.ply"):
                        for tensor in map_to_tensors.values():
                            value = tensor[i]
                            if tensor.dtype.kind == "f":
                                ply_file.write(np.float32(value).tobytes())
                            elif tensor.dtype == np.uint8:
                                ply_file.write(value.tobytes())
            else:
                # Transform the points to the original space
                import open3d as o3d
                import copy
                pcd = o3d.t.geometry.PointCloud(map_to_tensors)
                transform = np.asarray(db_outs.dataparser_transform)
                transform = np.concatenate([transform, np.array([[0, 0, 0, 1/db_outs.dataparser_scale]])], 0)
                transform = np.linalg.inv(transform)
                pcd_transformed = copy.deepcopy(pcd)
                pcd_transformed.transform(transform)
                p_min = pcd_transformed.get_min_bound().numpy().tolist()
                p_max = pcd_transformed.get_max_bound().numpy().tolist()
                crop_bounds = np.array([p_min[0],p_max[0],p_min[1],p_max[1],p_min[2],p_max[2]])
                o3d.t.io.write_point_cloud(splat_path, pcd_transformed)
            CONSOLE.print(f"[bold green]:white_check_mark: Saved splat at {splat_path}")
        else:
            CONSOLE.print(f"[bold green]Splat already exists at {splat_path}")
        
        # Read .ply file to add it uniform 3D grid
        ply_data = PlyData.read(splat_path)
        vertices = ply_data['vertex']
        verts = np.zeros(shape=[vertices.count, 3], dtype=np.float32)
        verts[:,0] = ply_data['vertex'].data['x']
        verts[:,1] = ply_data['vertex'].data['y']
        verts[:,2] = ply_data['vertex'].data['z']

        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        opacities = 1/(1 + np.exp(-vertices['opacity']))

        if self.ply_color_mode == 'sh_coeffs':
            rgbs = np.array([shs_3_to_rgb(v['f_dc_0'], v['f_dc_1'], v['f_dc_2']) for v in vertices])
            colors = np.vstack([rgbs[:,0], rgbs[:,1], rgbs[:,2]]).T # Red, Green, Blue
        elif self.ply_color_mode == 'rgb':
            colors = np.vstack([vertices['red']/255.0, vertices['green']/255.0, vertices['blue']/255.0]).T

        return verts, points, colors, opacities, crop_bounds

@dataclass
class ExportNeRFRGBDensity(Exporter):
    """
    Export RGB and density values from a NeRF model using camera view directions and spatial locations.
    
    Note: Only queries points within the specified scene bounding box.
    """
    nerf_model: Literal["nerfacto", "zipnerf", "tensorf", "splatfacto", "depth-nerfacto", "pynerf"] = "nerfacto"
    """Name of NeRF model"""
    scene_name: str = "ai_001_001"
    """Name of the optimized nerf scene"""
    dataset_type: Literal["hypersim", "scannet"] = "hypersim"
    """Name of the dataset which will be used."""
    dataset_path: str = r"E:\NeRF_datasets\hypersim_ngp_format"
    """The base path to the dataset scenes."""
    max_res: int = 128
    """The maximum resolution of the extracted features."""
    crop_scene: bool = True
    """Whether to crop the scene bounding box or not."""
    use_fixed_viewdirs: bool = False
    """Whether to use fixed view directions for sampling"""
    batch_size: int = 1048576
    """Batch size for number of rays to compute"""
    visualize: bool = False
    """Whether to plot the 3D feature grid or just save the rgbsigma in .npz file"""
    visualize_method: Literal["plotly", "pcd"] = "plotly"
    """Whether to plot 3D feature grid or point cloud. Only used when visualize=True"""
    show_poses: bool = False
    """Whether to plot camera positions. Only used when visualize=True"""
    show_boxes: bool = False
    """Whether to plot object bboxes. Only used when visualize=True"""
    min_vis_bbox: float = 0.2
    """Minimum length/extent of a object bounding box to visualize. Only used when show_boxes=True"""
    box_type: Literal["obb", "aabb"] = "obb"
    """Box type to visualize. Only used when show_boxes=True"""
    splat_color_mode: Literal["sh_coeffs", "rgb"] = "rgb"
    """If "rgb", export colors as r,g,b fields. Otherwise, export colors as shs coefficients for Splatfacto"""
    use_splat_o3d: bool = True
    """Whether to use Open3D to read/write/save the ply file for Splatfacto"""
    save_splat_w_boxes: bool = True
    """Whether to add bounding boxes to the splat file. Only used when save_splat_o3d=True & Splatfacto"""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        if str(self.load_config).endswith(".yml"):
            config_path = self.load_config
        else:
            config_path = ""
            if (self.dataset_type == "hypersim"):
                p = os.path.join(self.dataset_path, self.scene_name, "train")
            elif (self.dataset_type == "scannet"):
                p = os.path.join(self.dataset_path, self.scene_name)
            p = os.path.join(p, "nerf_data", self.nerf_model)
            if os.path.exists(p):
                p = glob.glob(os.path.join(p, '*'))
                if len(p) > 1:
                    print(f"Found {len(p)} configurations")
                for f in p:
                    if os.path.exists(os.path.join(f, 'nerfstudio_models')):
                        config_path = Path(os.path.join(f,"config.yml"))
            else:
                print(f"No trained model found in {p}")
                sys.exit(1)
            if (config_path == ""):
                print(f"Found config, but no checkpoint found for {self.nerf_model} in {self.scene_name}")
                sys.exit(1)

        conf, pipeline, ckpt_path, _ = eval_setup(config_path)
        self.scene_name =  os.path.basename(conf.pipeline.datamanager.data)

        if (self.dataset_type == "hypersim") and (self.scene_name == "train"):
            p = os.path.normpath(conf.pipeline.datamanager.data)
            self.scene_name = p.split(os.sep)[-2]
            scene_dir = os.path.join(self.dataset_path, self.scene_name, 'train')
            self.transforms_filename = "transforms.json"

        elif (self.dataset_type == "scannet"):
            scene_dir = os.path.join(self.dataset_path, self.scene_name)
            self.transforms_filename = "obj_instances.json"
        
        json_path = os.path.join(scene_dir, self.transforms_filename)
        out_path = os.path.join(self.output_dir, f'{self.scene_name}.npz')

        if not os.path.exists(json_path):
            CONSOLE.print(f"[bold yellow]{self.transforms_filename} not found in {self.scene_name}. Exiting.")
            sys.exit(1)

        if (self.nerf_model == "nerfacto"):
            assert isinstance(pipeline.model, NerfactoModel)
            model: NerfactoModel = pipeline.model
        elif (self.nerf_model == "zipnerf"):
            from zipnerf_ns.zipnerf_model import ZipNerfModel 
            sys.path.append(r"C:\Users\OEM\nerf-gs-detect\nerfstudio\zipnerf-pytorch") 
            assert isinstance(pipeline.model, ZipNerfModel)
            model: ZipNerfModel = pipeline.model
        elif (self.nerf_model == "tensorf"):
            assert isinstance(pipeline.model, TensoRFModel)
            model: TensoRFModel = pipeline.model
        elif (self.nerf_model == "splatfacto"):
            assert isinstance(pipeline.model, SplatfactoModel)
            model: SplatfactoModel = pipeline.model
            # TODO: Implement Splatfacto
        elif (self.nerf_model == "depth-nerfacto"):
            assert isinstance(pipeline.model, DepthNerfactoModel)
            model: DepthNerfactoModel = pipeline.model
        elif (self.nerf_model == "pynerf"):
            from pynerf.pynerf.models.pynerf_model import PyNeRFModel 
            sys.path.append(r"C:\Users\OEM\nerf-gs-detect\nerfstudio\pynerf")
            assert isinstance(pipeline.model, PyNeRFModel)
            model: PyNeRFModel = pipeline.model
        else:
            CONSOLE.print(f"[bold yellow]Invalid NeRF model: {self.nerf_model}. Exiting.")
            sys.exit(1)

        query_nerf_model(
            self.nerf_model, model, pipeline, json_path, out_path, conf,
            self.dataset_type, self.max_res, self.batch_size,self.min_vis_bbox, 
            self.crop_scene, self.use_fixed_viewdirs, self.visualize,
            self.show_poses, self.show_boxes, self.visualize_method, 
            self.box_type, self.splat_color_mode, self.use_splat_o3d,
            self.save_splat_w_boxes
        )

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportNeRFRGBDensity, tyro.conf.subcommand(name="nerf-rgbd")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa

