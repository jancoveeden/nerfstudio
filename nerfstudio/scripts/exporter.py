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
from pynerf.pynerf.models.pynerf_model import PyNeRFModel

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

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

    def __init__(self, grid_limits, max_res, device="cpu", nerf_name="nerfacto") -> None:
        self.grid_limits =  grid_limits
        self.max_res = max_res
        self.device = device
        self.nerf_name = nerf_name

        max_pt = torch.Tensor(self.grid_limits[1::2])
        min_pt = torch.Tensor(self.grid_limits[::2])
        scale_factor = (self.max_res - 0.) / torch.max(max_pt - min_pt)
        max_xyz = ((max_pt - min_pt) * scale_factor + 0.).to(torch.int64)
        self.grid = torch.zeros(4, max_xyz[0], max_xyz[1], max_xyz[2], dtype=torch.float32, device=device)
        self.max_xyz = max_xyz

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
        Adds RGB and Alpha values to an existing 3D feature grid

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

    def get_box_corners_edges(self, obb):
        """
        Compute 8 corners and 12 edges
        
        position: center of obb
        extents: size of box in each direction from its center to outer surface xyz
        orientation: rotation matrix
        """
        # Sanity Check: OrientedBox(R=orientation, T=position, S=extents)
        orientation = np.array(obb.R)
        position = np.array(obb.T)
        extents = np.array(obb.S)

        extents = extents * self.scene_scale * self.bbox_scale 
        position = position * self.scene_scale * self.bbox_scale

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
        corners = np.dot(corners, orientation.T) + position
        corners = (corners - self.grid_limits[::2]) / (self.grid_limits[1::2] - self.grid_limits[::2])
        corners = (torch.Tensor(corners) * (self.max_xyz - 1)) 
        corners = np.array(corners)

        edge_planes = [[0, 1], [1, 2], [2, 3], [3, 0],  # bottom edges
                       [4, 5], [5, 6], [6, 7], [7, 4],  # top edges
                       [0, 4], [1, 5], [2, 6], [3, 7]]  # connecting edges

        return corners, edge_planes

    def plot_point_cloud(self, alpha_threshold=0.5):
        """
        Visualizes the extracted feature grid as point cloud

        Args:
            alpha_threshold: Density threshold at which points are saved
            save_image: Whether to save an image or not.
        """
        import open3d as o3d
        grid = self.grid.cpu().detach().numpy()

        W, L, H = grid.shape[1:]
        x, y, z = np.mgrid[0:W, 0:L, 0:H]

        alphas = grid[3].flatten()
        mask = alphas > alpha_threshold
        alphas = alphas[mask]

        x, y, z = x.flatten()[mask], y.flatten()[mask], z.flatten()[mask]

        rgb = grid[:3, :, :, :].reshape(3, -1).T 
        rgb = rgb[mask]

        pcd = o3d.geometry.PointCloud()
        points = np.vstack((x, y, z)).T
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def visualize_depth_grid(self, db_outs, obbs, scene_scale=0.3333, bbox_scale=0.5, show_poses=False, show_boxes=False):
        """
        Visualizes the extracted feature grid as depth grid

        Args:
            db_outs: Nerfstudio pipeline.datamanager.train_dataparser_outputs object
            obbs: List of OrientedBoxes for each object
            show_poses: Whether to plot the camera poses in the grid
            show_boxes: Whether to plot the object bounding boxes in the grid
        """
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
            self.scene_scale = scene_scale
            self.bbox_scale = bbox_scale 
            bbox_min = self.grid_limits[::2]
            bbox_max = self.grid_limits[1::2]
            # bbox_min = np.array([-1., -1., -1.])
            # bbox_max = np.array([1., 1., 1.])

            for i, obb in enumerate(obbs):
                pos = np.array(obb.T) * bbox_scale * scene_scale
                if ((pos < bbox_min).any() or (pos > bbox_max).any()):
                    continue
                vertices, edges = self.get_box_corners_edges(obb)

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

def get_ngp_obj_bounding_box(xform, extent):
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

def estimate_scene_box(json_dict, nerf_name, dataset_type, margin=0.2):
    """
    Estimates scene bounding box using 
        - Object bounding boxes
        - Cameras
    
    Returns:
        min_pt: 1x3 matrix
        max_pt: 1x3 matrix
    """
    scales = {"nerfacto": 1.0, "depth-nerfacto": 1.0, "tensorf": 1.0, "zipnerf": 2.0, "splatfacto": 2.0}

    if (dataset_type == 'scannet'):
        mini = [ins['min_pt'] for ins in json_dict['instances']]
        maxi = [ins['max_pt'] for ins in json_dict['instances']]

        min_pts = torch.tensor(mini)
        max_pts = torch.tensor(maxi)

        min_pt = min_pts.min(dim=0)[0]
        max_pt = max_pts.max(dim=0)[0]

        CONSOLE.print(f"[bold magenta]Estimated scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")

        # Re-center scenebox and scale to [-1, +1] except for zipnerf [-2, +2]
        scale_f = scales.get(nerf_name, 1.0) / torch.max(max_pt - min_pt)
        center = (min_pt + max_pt) / 2.0
        z_scale = scale_f + (margin / torch.max(max_pt + min_pt))

        if (min_pt[0] < np.abs(center[0]) + min_pt[0]) and (max_pt[0] > np.abs(center[0]) - max_pt[0]):
            min_pt[0] = -8.0
            max_pt[0] = 8.0
        
        if (min_pt[1] < np.abs(center[1]) + min_pt[1]) and (max_pt[1] > np.abs(center[1]) - max_pt[1]):
            min_pt[1] = -8.0
            max_pt[1] = 8.0

        if (min_pt[2] < np.abs(center[2]) + min_pt[2]) and (max_pt[2] > np.abs(center[2]) - max_pt[2]):
            min_pt[2] = -np.abs(center[2])
            max_pt[2] = np.abs(center[2])

        for i in range(3):
            sign = 1 if center[i] <= 0 else -1
            scale = z_scale if i == 2 else scale_f
            min_pt[i] = (min_pt[i] + sign * center[i]) * scale
            max_pt[i] = (max_pt[i] - sign * center[i]) * scale
        
    elif (dataset_type == 'hypersim'):
        min_box_pt = []
        max_box_pt = []
        min_cams_pt = []
        max_cams_pt = []

        # Get min & max corners from bbox:
        for obj in json_dict['bounding_boxes']:
            extent = np.array(obj['extents'])
            orientation = np.array(obj['orientation'])
            position = np.array(obj['position'])

            xform = np.hstack([orientation, np.expand_dims(position, 1)])
            min_pt_, max_pt_ = get_ngp_obj_bounding_box(xform, extent)
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
        
        # Re-center scenebox and scale to [-1, +1] except for zipnerf
        scale_f = scales.get(nerf_name, 1.0) / torch.max(max_pt - min_pt)
        center = (min_pt + max_pt) / 2.0
        z_scale = scale_f + (margin / torch.max(max_pt - min_pt))

        for i in range(3):
            sign = 1 if center[i] <= 0 else -1
            scale = z_scale if i == 2 else scale_f
            min_pt[i] = (min_pt[i] + sign * center[i]) * scale
            max_pt[i] = (max_pt[i] - sign * center[i]) * scale
    

    min_pt = torch.clamp(min_pt, min=-scales.get(nerf_name, 1.0), max=scales.get(nerf_name, 1.0))
    max_pt = torch.clamp(max_pt, min=-scales.get(nerf_name, 1.0), max=scales.get(nerf_name, 1.0))

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

def parse_transforms_to_obbs(bbox_dict, max_xyz, dataset_type='hypersim', min_extent=0.2):
    """
    Converts the extents, orientation and position in the 
    transform.json/obj_instances.json to oriented bounding boxes (OBBs)

    Args:
        bbox_dict: json_dict
    """
    res = max_xyz.cpu().numpy()
    obbs = []

    if (dataset_type == 'hypersim'):
        bboxes = bbox_dict["bounding_boxes"]
        for obj in bboxes:
            extents = np.array(obj['extents'])
            orientation = np.array(obj['orientation'])
            position = np.array(obj['position'])

            if (extents < min_extent).any(): # Filters out small objects
                continue

            obb = OrientedBox(R=orientation, T=position, S=extents)
            obbs.append(obb)
    elif (dataset_type == 'scannet'):
        bboxes = bbox_dict["instances"]

        obb = np.array([x['obb'] for x in bbox_dict['instances']])
        max_pt = np.array([x['max_pt'] for x in bbox_dict['instances']])
        min_pt = np.array([x['min_pt'] for x in bbox_dict['instances']])
        labels = [x['label'] for x in bbox_dict['instances']]
        bbox_min = np.min(min_pt, axis=0)
        bbox_max = np.max(max_pt, axis=0)

        # obb[:, 3:6] = (obb[:, 3:6] / (bbox_max - bbox_min)) * 2.0
        # obb[:, :3] = ((obb[:, :3] - bbox_min) / (bbox_max - bbox_min)) * 2.0

        # keep_labels = ['bed', 'desk', 'chair']
        # keep = np.zeros(len(obb), dtype=bool)
        # for i in range(len(obb)):
        #     if labels[i] in keep_labels:
        #         keep[i] = True
        #     elif np.min(obb[i, 3:6]) < min_extent:
        #         keep[i] = False
        # obb = obb[keep]

        for ob in obb:
            position = ob[:3]
            extents = ob[3:6] 
            angle = ob[6]

            position[0] -= 4.0
            position[1] -= 4.0
            position[2] -= 0.8

            orientation = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,              0,             1]
            ])

            if np.min(extents) < min_extent: # Filters out small objects
                continue
            
            obb = OrientedBox(R=orientation, T=position, S=extents)
            obbs.append(obb)
    
    return obbs

def extract_splatfacto(model, sampler, coords, cams):
    """
    [INCOMPLETE]
    Can be used to visualize 3D gaussians in plotly
    """
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

def compute_pixel_area(dir, nerf_model):
    # dx = torch.sqrt(torch.sum((directions - start) ** 2, dim=-1)) 
    # dy = torch.sqrt(torch.sum((directions - end) ** 2, dim=-1)) 
    # pixel_area = (dx * dy)[..., None] # ("num_rays":..., 1)
    # return pixel_area

    pixel_area = None
    if (nerf_model == 'pynerf'):
        # Simulating neighbouring pixel ray directions
        delta = 1e-5
        dx = torch.sqrt(torch.sum((dir + delta - dir) ** 2, dim=-1))
        dy = torch.sqrt(torch.sum((dir + delta - dir) ** 2, dim=-1))
        pixel_area = (dx * dy)[..., None] # ("num_rays":..., 1)
    
    return pixel_area

def query_nerf_model(nerf_name, model, pipeline, json_path, output_path, dataset_type='hypersim', 
                    max_res=128, batch_size=4096, min_bbox=0.2, crop_scene=True, use_fixed_viewdirs=False, 
                    visualize=False, show_poses=False, show_boxes=False, vis_method="plotly"):
    """
    Extracts rgbsigma from a given pre-trained NeRF scene

    Args:
        nerf_name: Name of NeRF model used for novel view synthesis
        model: NeRF model object (Nerfacto, Zip-NeRF, TensoRF, Depth-Nerfacto)
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
    """
    device = model.device
    db_outs = pipeline.datamanager.train_dataparser_outputs
    scene_scale = pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    cams = db_outs.cameras
    pcd_thresh = 0.5
    trf = nerf_name == "tensorf"
    zp = nerf_name == "zipnerf"

    with open(json_path) as f:
        json_dict = json.load(f) 
    
    # Get estimated nerfstudio scene bbox or predefined scene bbox
    if (crop_scene):
        min_pt, max_pt = estimate_scene_box(json_dict, nerf_name, dataset_type)
    else:
        min_pt = db_outs.scene_box.aabb[0] #- 1.0
        max_pt = db_outs.scene_box.aabb[1] #+ 1.0

    # Generate [4, res_x, res_y, res_z] grid and xyz coordinates
    grid_limits = np.array([min_pt[0],max_pt[0],min_pt[1],max_pt[1],min_pt[2],max_pt[2]])
    grid_sampler = Grid_Sampler(grid_limits, max_res, device="cpu", nerf_name=nerf_name) 
    grid_coords = grid_sampler.generate_coords()
    coords_to_render = grid_coords.view(-1, 3) # [res_x*res_y*res_z, 3] 

    # Parse bboxes and get scene + bbox scale
    obbs = parse_transforms_to_obbs(json_dict, grid_sampler.max_xyz, dataset_type, min_bbox)
    if dataset_type == 'hypersim':
        bbox_scale = 2.0 * (1/float(json_dict["aabb_scale"])) # Reverse of 0.5 * meta.get("aabb_scale", 1)
        bbox_scale *= 2.0 if zp else 1.0
    elif dataset_type == 'scannet':
        bbox_scale = 0.7
        print(bbox_scale)
        print(scene_scale)
        print(bbox_scale * scene_scale)
    
    # Enables SceneBox positions for Nerfacto
    if (nerf_name in ['nerfacto', 'depth-nerfacto', 'pynerf']):
        model.field.spatial_distortion = None 

    # Crop scene boundaries
    aabb_lengths = max_pt - min_pt
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

    if (dataset_type == 'scannet'):
        camera_views = camera_views[1::2]
    #CONSOLE.print(f"[bold blue]Number of dataset images: {len(db_outs.image_filenames)}")
    CONSOLE.print(f"[bold blue]Batch size set to: {batch_size}")
    CONSOLE.print(f"[bold blue]Number of views to extract from: {len(camera_views)}")
    CONSOLE.print(f"[bold magenta]Using scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")

    if (nerf_name == "splatfacto"):
        grid_sampler = extract_splatfacto(model, grid_sampler, coords_to_render, cams)
        pcd_thresh = 0.2

    # Extract RGB and density into the 3D feature grid
    if not (nerf_name == "splatfacto"):
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
            grid_sampler.visualize_depth_grid(db_outs, obbs, scene_scale, bbox_scale, show_poses, show_boxes)
        else:
            grid_sampler.plot_point_cloud(pcd_thresh)
    else:
        # Save rgbsigma
        CONSOLE.print(f"[bold green]Successfully extracted rgbsigma from {nerf_name}")
        rgbsigma = grid_sampler.get_nerf_rpn_output()
        min_pt = min_pt.cpu().numpy()
        max_pt = max_pt.cpu().numpy()
        res = grid_sampler.max_xyz.cpu().tolist()
        CONSOLE.print(f"[bold green]Feature grid resolution: {grid_sampler.grid.size()}")
        CONSOLE.print(f"[bold green]Saving rgbsigma of size: {rgbsigma.shape}")
        np.savez_compressed(output_path, rgbsigma=rgbsigma, resolution=res,
                            bbox_min=min_pt, bbox_max=max_pt,
                            scale=scene_scale, offset=0.0)
        

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
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
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
        if not all(len(tensor) == count for tensor in map_to_tensors.values()):
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

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        filename = self.output_dir / "splat.ply"

        count = 0
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

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

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

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)

@dataclass
class ExportNeRFRGBDensity(Exporter):
    """
    Export RGB and density values from a NeRF model using camera view directions and spatial locations.
    
    Note: Only queries points within the specified scene bounding box.
    """
    nerf_model: Literal["nerfacto", "zipnerf", "tensorf", "splatfacto", "depth-nerfacto", "pynerf"] = "nerfacto"
    """Name of NeRF model"""
    scene_name: str = "ai_001_001"
    """Name of the pre-trained scene"""
    dataset_type: Literal["hypersim", "scannet"] = "hypersim"
    """Name of the dataset which will be used. TODO: Update for future datasets."""
    dataset_path: str = "hypersim"
    """The path to the scenes in instant-ngp data format."""
    transforms_filename: str = "transforms.json"
    """The name of the transforms file containing camera metadata, camera poses and bounding boxes."""
    ckpt_name: str = "step-000006999.ckpt"
    """Name of the snapshot/checkpoint. Currently unused."""
    max_res: int = 128
    """The maximum resolution of the extracted features."""
    crop_scene: bool = True
    """Whether to crop the scene bounding box or not."""
    use_fixed_viewdirs: bool = False
    """Whether to use fixed view directions for sampling"""
    batch_size: int = 4096
    """Batch size for number of rays to compute"""
    visualize: bool = False
    """Whether to plot the 3D feature grid or just save the rgbsigma in .npz file"""
    visualize_method: Literal["plotly", "pcd"] = "plotly"
    """Whether to plot 3D feature grid or point cloud. Only used when visualize=True"""
    show_poses: bool = False
    """Whether to plot camera positions in plotly visualization"""
    show_boxes: bool = False
    """Whether to plot object bboxes in plotly visualization"""
    min_bbox: float = 0.2
    """Minimum length/extent of a object bounding box to visualize. Only used when show_boxes=True"""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        conf, pipeline, ckpt_path, _ = eval_setup(self.load_config)
        self.scene_name =  os.path.basename(conf.pipeline.datamanager.data)

        if (self.dataset_type == "hypersim") and (self.scene_name == "train"):
            p = os.path.normpath(conf.pipeline.datamanager.data)
            self.scene_name = p.split(os.sep)[-2]
            scene_dir = os.path.join(self.dataset_path, self.scene_name, 'train')
            self.transforms_filename = "transforms.json"

        elif (self.dataset_type == "scannet"):
            scene_dir = os.path.join(self.dataset_path, self.scene_name)
            # self.transforms_filename = "obj_instances.json"
            self.transforms_filename = "bboxes.json"
        
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
            assert isinstance(pipeline.model, PyNeRFModel)
            model: PyNeRFModel = pipeline.model
        else:
            CONSOLE.print(f"[bold yellow]Invalid NeRF model: {self.nerf_model}. Exiting.")
            sys.exit(1)

        query_nerf_model(
            self.nerf_model, model, pipeline, json_path, out_path, 
            self.dataset_type, self.max_res, self.batch_size,self.min_bbox, 
            self.crop_scene, self.use_fixed_viewdirs, self.visualize,
            self.show_poses, self.show_boxes, self.visualize_method
        )

        if not (self.visualize):
            CONSOLE.print(f"[bold green]:white_check_mark: Saved features of extracted scene: {self.scene_name} ")


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

