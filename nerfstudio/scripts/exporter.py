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
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal
from tqdm import tqdm

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
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.field_components.field_heads import FieldHeadNames
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

class Grid_Sampler:
    """
    Grid sampler to represent a scene as a grid of voxels.

    grid contains 7-channel 3D grid
        - Red ------> grid[0, :, : ,:]  --|
        - Green ----> grid[1, :, : ,:]  --|--|-- rgbsigma
        - Blue -----> grid[2, :, : ,:]  --|--|
        - Alpha ----> grid[3, :, : ,:]  --|
        - x_coords -> grid[4, :, : ,:]
        - y_coords -> grid[5, :, : ,:]
        - z_coords -> grid[6, :, : ,:]

    Args:
        grid_limits: Scene box xyz limits as an list -> (x_min, x_max, y_min, y_max, z_min, z_max)
        max_res: int: Maximum resolution for x, y, z-axis
        device: Device where the grid must be stored
    """

    def __init__(self, grid_limits, max_res, device="cpu") -> None:
        self.grid_limits =  grid_limits
        self.max_res = max_res
        self.device = device

        self.grid = torch.zeros(7, int(max_res), int(max_res), int(max_res), dtype=torch.float32, device=device)

        # Unused
        self.delta = 1e-2
        self.grid_step = (self.grid_limits[1] - self.grid_limits[0]) / (max_res - 1)
        self.voxel_grid_step = (self.grid_limits[1] - self.grid_limits[0]) / max_res

    def density_to_alpha(self, density):
        """
        Converts queried density to alpha values

        Args:
            density: [batch_size, 1]
        """
        # return torch.clip(1. - torch.exp(-1e-2 * density), 0., 1.)            # NeRAF
        return np.clip(1.0 - torch.exp(-torch.exp(density) / 100.0), 0.0, 1.0)  # NeRF-RPN instant-ngp
        #return torch.clip(1.0 - torch.exp(-density * 1e-2), 0.0, 1.0)
        

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

        alpha = self.density_to_alpha(density)  # [batch_size, 1]
        colour = torch.sigmoid(rgb)             # [batch_size, 3]

        rays_xyz = (rays_xyz - self.grid_limits[::2]) / (self.grid_limits[1::2] - self.grid_limits[::2])
        idxs = (rays_xyz * (self.max_res - 1)).round().int()
        x_idxs, y_idxs, z_idxs = idxs[:, 0], idxs[:, 1], idxs[:, 2]

        mask = (x_idxs >= 0) & (x_idxs < self.grid.shape[1]) & \
               (y_idxs >= 0) & (y_idxs < self.grid.shape[2]) & \
               (z_idxs >= 0) & (z_idxs < self.grid.shape[3])
        
        x_idxs = x_idxs[mask]
        y_idxs = y_idxs[mask]
        z_idxs = z_idxs[mask]

        colour = colour[mask].float()
        alpha = alpha[mask].float()

        self.grid[0, x_idxs, y_idxs, z_idxs] = colour[:, 0].squeeze() # [batch_size]
        self.grid[1, x_idxs, y_idxs, z_idxs] = colour[:, 1].squeeze() # [batch_size]
        self.grid[2, x_idxs, y_idxs, z_idxs] = colour[:, 2].squeeze() # [batch_size]
        self.grid[3, x_idxs, y_idxs, z_idxs] = alpha.squeeze()        # [batch_size]
    
    def generate_voxel_coords(self):
        """
        Generates xyz coordinates for a voxel grid with voxel_grid_step step size
        """
        grid_v_coords = torch.meshgrid(torch.arange(self.grid_limits[0], self.grid_limits[1], self.voxel_grid_step), 
                                    torch.arange(self.grid_limits[2], self.grid_limits[3], self.voxel_grid_step), 
                                    torch.arange(self.grid_limits[4], self.grid_limits[5], self.voxel_grid_step), 
                                    indexing='ij')
        grid_v_coords = torch.stack(grid_v_coords, dim=0).detach().cpu()

        return grid_v_coords

    def generate_coords(self):
        """
        Generates xyz coordinates for a point grid
        """
        x = torch.linspace(self.grid_limits[0], self.grid_limits[1], self.max_res)
        y = torch.linspace(self.grid_limits[2], self.grid_limits[3], self.max_res)
        z = torch.linspace(self.grid_limits[4], self.grid_limits[5], self.max_res)
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        grid_coords = torch.stack([x, y, z], dim=-1).reshape(3, self.max_res,self.max_res,self.max_res).cpu()

        return grid_coords

    def get_nerf_rpn_output(self):
        """
        Returns rgbsigma in the NeRF-RPN format
        Shape: (res_x * res_y * res_z, 4)
        Refer to https://github.com/lyclyc52/NeRF_RPN
        """
        rgbsigma = self.grid[:4,:,:,:]
        rgbsigma = rgbsigma.view(-1, 4) # (W * L * H, 4) = (res_x * res_y * res_z, 4)

        return rgbsigma
    
    def plot_point_grid(self):
        grid = self.grid.cpu().detach().numpy()
        r = grid[0]
        g = grid[1]
        b = grid[2]

        rgb = np.stack((r, g, b), axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        
        depth, height, width, _ = rgb.shape
        x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colours = rgb.reshape(-1, 3) / 255.0
        ax.scatter(x, y, z, c=colours, marker='.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    
    def visualize_depth_grid(self, db_outs, show_poses=False):
        """
        Visualizes the extracted feature grid

        Args:
            db_outs: Nerfstudio pipeline.datamanager.train_dataparser_outputs object
            show_poses: Whether to plot the camera poses in the grid
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

        # Camera Positions
        if (show_poses):
            camera_positions = []
            for cam in db_outs.cameras:
                pose = cam.camera_to_worlds.cpu().numpy()
                camera_pos = pose[:3, 3] / 3.
                camera_positions.append(camera_pos)
            camera_positions = torch.Tensor(np.array(camera_positions))
            cam_poses = (camera_positions - self.grid_limits[::2]) / (self.grid_limits[1::2] - self.grid_limits[::2])
            cam_poses = (cam_poses * (self.max_res - 1)).round().int()

            # Add camera positions as dots
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

def nerf_matrix_to_ngp(nerf_matrix, scale=0.33, offset=[0.5, 0.5, 0.5]):
    """
    Convert a matrix from the NeRF coordinate system to the instant-ngp coordinate system.
    """
    ngp_matrix = np.copy(nerf_matrix)
    ngp_matrix[:, 1:3] *= -1
    ngp_matrix[:, -1] = ngp_matrix[:, -1] * scale + offset

    # Convert xyz<-yzx
    tmp = np.copy(ngp_matrix[0, :])
    ngp_matrix[0, :] = ngp_matrix[1, :]
    ngp_matrix[1, :] = ngp_matrix[2, :]
    ngp_matrix[2, :] = tmp

    return ngp_matrix

def get_ngp_obj_bounding_box(xform, extent):
    """
    Get AABB from the OBB of an object in ngp coordinates.
    
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

def get_scene_bounding_box(json_dict, margin=0.1):
    """
    Estimates scene bounding box using object bounding boxes
    
    Returns:
        min_pt: 1x3 matrix
        max_pt: 1x3 matrix
        ngp_min_pt: 1x3 matrix
        ngp_max_pt: 1x3 matrix
    """
    min_pt = []
    max_pt = []

    # Get min & max corners for each bbox:
    for obj in json_dict['bounding_boxes']:
        extent = np.array(obj['extents'])
        orientation = np.array(obj['orientation'])
        position = np.array(obj['position'])

        xform = np.hstack([orientation, np.expand_dims(position, 1)]) # 3x4 transformation matrix
        min_pt_, max_pt_ = get_ngp_obj_bounding_box(xform, extent)
        
        min_pt.append(min_pt_)
        max_pt.append(max_pt_)

    min_pt = np.array(min_pt) 
    max_pt = np.array(max_pt)
    min_pt = np.min(min_pt, axis=0) # 1x3
    max_pt = np.max(max_pt, axis=0) # 1x3

    # Slightly enlarge scene bbox
    added_pnts = (max_pt - min_pt) * margin
    min_pt -= added_pnts
    max_pt += added_pnts

    # Get min/max corners in ngp format
    xform = np.hstack([np.eye(3, 3), np.expand_dims(min_pt, 1)]) # 3x4
    ngp_min_pt = nerf_matrix_to_ngp(xform)[:, 3] # 1x3
    xform = np.hstack([np.eye(3, 3), np.expand_dims(max_pt, 1)])  # 3x4
    ngp_max_pt = nerf_matrix_to_ngp(xform)[:, 3] # 1x3

    min_pt = torch.tensor(min_pt)
    max_pt = torch.tensor(max_pt) 

    return min_pt, max_pt, ngp_min_pt, ngp_max_pt

def generate_fixed_viewdirs():
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
    viewdirs = torch.stack(viewdirs, dim=0)
    return viewdirs

def query_nerf_model(nerf_name, model, pipeline, json_path, output_path, 
                    dataset_type='hypersim', max_res=256, crop_bbox=True, 
                    use_fixed_viewdirs=False, batch_size=4096, visualize=False):
    """
    Extract rgbsigma from a given pre-trained NeRF scene

    Args:
        nerf_name: Name of NeRF model used for novel view synthesis
        model: NeRF model object (Splatfacto, Nerfacto or ZipNeRF)
        pipeline: NeRF model pipeline module
        json_path: Path to transform.json containing bboxes, transformation matrices etc.
        output_path: Path to .npz file to save
        dataset_type: Dataset name (Hypersim)
        max_res: Maximum resolution for an axis for the feature grid 
        crop_bbox: Whether to crop the scene bounding box or not.
    """
    device = model.device
    db_outs = pipeline.datamanager.train_dataparser_outputs

    # Get bounding boxes:
    with open(json_path) as f:
        json_dict = json.load(f)
        if "bounding_boxes" in json_dict and len(json_dict['bounding_boxes']) > 0:
            bounding_boxes = json_dict["bounding_boxes"]
        else:
            CONSOLE.print("[bold red]Bounding boxes in transforms.json not found. Exiting.")
            sys.exit(1)

    ### Cameras
    cams = db_outs.cameras

    ### Get estimated nerfstudio scene bbox and actual scene bbox
    if dataset_type == 'hypersim':
        #min_pt, max_pt, ngp_min_pt, ngp_max_pt = get_scene_bounding_box(json_dict)
        #CONSOLE.print(f"[bold magenta]Estimated scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")
        
        scene_bbox = db_outs.scene_box
        min_pt = scene_bbox.aabb[0] # minimum (x,y,z) point
        max_pt = scene_bbox.aabb[1] # maximum (x,y,z) point
        CONSOLE.print(f"[bold magenta]Nerfstudio scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")

        if (crop_bbox):
            min_pt = torch.Tensor([-0.95, -0.95, -0.95])
            max_pt = torch.Tensor([0.95, 0.95, 0.95])
            CONSOLE.print(f"[bold magenta]Cropped scene bbox: \nmin_pt: {min_pt}\nmax_pt: {max_pt}")
    else:
        CONSOLE.print(f"[bold yellow]Unknown dataset type: {dataset_type}. Exiting.")
        sys.exit(1) 

    CONSOLE.print(f"[bold blue]Number of training images/views: {len(db_outs.image_filenames)}")

    ### Get resolution of 3D feature grid
    res = (max_pt - min_pt) / (max_pt - min_pt).max() * max_res
    res = res.round().int().tolist()
    res_x, res_y, res_z = res

    ### Extract RGB and density into the feature grid
    if (nerf_name == "zipnerf"):
        from extract import get_rgbsigma
        
        # Create 3D feature grid
        x = torch.linspace(min_pt[0], max_pt[0], res_x)
        y = torch.linspace(min_pt[1], max_pt[1], res_y)
        z = torch.linspace(min_pt[2], max_pt[2], res_z)

        z, y, x = torch.meshgrid(z, y, x, indexing="ij") # e.g. x size: [128, 128, 128]  
        xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3).unsqueeze(0).to(device) # e.g. size: [1, 2097152, 3]
        # xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device) # e.g. size: [2097152, 3]
        rgb_mean = torch.zeros((res_x * res_y * res_z, 3)).to(device) # e.g. size: [2097152, 3]

        CONSOLE.print(f"[bold blue]Extracting {nerf_name} with resolution: {res}")
        CONSOLE.print(f"[bold blue]Iterating through {cams.size} cameras...")
        for cam_idx in tqdm(range(cams.size), desc="Extracting RGB/Sigma"):
            trans_mat = cams[cam_idx].camera_to_worlds
            viewdir = torch.Tensor(torch.Tensor(trans_mat[:3, :3]).unsqueeze(0).to(device) @ torch.Tensor([0, 0, -1]).to(device))

            rgb, sigma = get_rgbsigma(model, xyz, viewdir, batch_size)
            rgb_mean += rgb 

        rgb_mean = rgb_mean / len(range(cams.size))
        rgbsigma = torch.cat([rgb_mean, sigma], dim=1) # (res_z * res_y * res_x, 4)

        # grid = rgbsigma.view(4, max_res, max_res, max_res).cpu().numpy()
        # x, y, z = np.mgrid[0:grid.shape[1], 0:grid.shape[2], 0:grid.shape[3]]
        # fig = go.Figure(data=go.Volume(
        #     x=x.flatten(),
        #     y=y.flatten(),
        #     z=z.flatten(),
        #     value=grid[3].flatten(), 
        #     opacity=0.1,
        #     surface_count=17,
        #     colorscale='Viridis'
        # ))
        # fig.show()
        # exit()

        rgbsigma = rgbsigma.cpu().numpy()
        min_pt = min_pt.cpu().numpy()
        max_pt = max_pt.cpu().numpy()
        CONSOLE.print(f"[bold green]Saving rgbsigma of size: {rgbsigma.shape}")
        np.savez_compressed(output_path, rgbsigma=rgbsigma, resolution=res,
                            bbox_min=min_pt, bbox_max=max_pt,
                            scale=0.3333, offset=0.0)
    
    # elif (nerf_name == "zipnerf_"):

        
    elif (nerf_name == "nerfacto"):
        grid_limits = np.array([min_pt[0],max_pt[0],min_pt[1],max_pt[1],min_pt[2],max_pt[2]])
        grid_sampler = Grid_Sampler(grid_limits, max_res, device="cpu") # [7, max_res, max_res, max_res]
        grid_coords = grid_sampler.generate_coords()
        grid_sampler.grid[4:,:,:,:] = grid_coords
        coords_to_render = grid_coords.view(-1, 3) # [max_res*max_res*max_res, 3]

        vision_field = model.field
        spatial_distort = vision_field.spatial_distortion
        vision_field.spatial_distortion = None # Enables SceneBox positions
        aabb_lengths = max_pt - min_pt
        CONSOLE.print(f"[bold blue]Batch size set to: {batch_size}")

        with torch.no_grad():
            with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TimeElapsedColumn(),
                        MofNCompleteColumn(),
                        transient=True,
            ) as progress:
                task = progress.add_task("[green]Extracting feature grid...", total=coords_to_render.shape[0])
                for i in range(0, coords_to_render.shape[0], batch_size):
                    if i + batch_size > coords_to_render.shape[0]:
                        batch_size = coords_to_render.shape[0] - i

                    batch_coords = coords_to_render[i:i+batch_size] # [batch_size, 3]
                    ori = (batch_coords * aabb_lengths) #+ min_pt   # [batch_size, 3]
                    ori = (batch_coords * aabb_lengths)             # [batch_size, 3]
                    start = torch.zeros_like(ori)
                    end = torch.zeros_like(ori)

                    if not (use_fixed_viewdirs):
                        rgbs = []
                        densities = []    
                        for j in range(cams.size):
                            trans_mat = cams[j].camera_to_worlds
                            viewdir = torch.Tensor(torch.Tensor(trans_mat[:3, :3]) @ torch.Tensor([0., 0., -1.]))
                            
                            dir = viewdir.expand(batch_size, -1)
                            f = Frustums(ori, dir, start, end, None)
                            cam_indices = torch.zeros((batch_size,1), dtype=torch.int)
                            rays = RaySamples(f, cam_indices).to(model.device)
        
                            field_outputs = vision_field.forward(rays)

                            rgb = field_outputs[FieldHeadNames.RGB]
                            density = field_outputs[FieldHeadNames.DENSITY]

                            ## Renderer
                            # rgb = rgb.unsqueeze(1)
                            # density = density.unsqueeze(1)
                            # weights = torch.ones_like(density)
                            # rgb = model.renderer_rgb(rgb=rgb, weights=weights)

                            rgbs.append(rgb.cpu())
                            densities.append(density.cpu())

                        rgb = torch.mean(torch.stack(rgbs, dim=0), dim=0)          # [batch_size, 3]
                        density = torch.mean(torch.stack(densities, dim=0), dim=0) # [batch_size, 1]

                    else: # Generate fixed view directions...
                        fixed_views = generate_fixed_viewdirs()
                        rgbs = []
                        densities = []    
                        for j in range(len(fixed_views)):
                            dir = fixed_views[j].expand(batch_size, -1)
                            f = Frustums(ori, dir, start, end, None)
                            cam_indices = torch.zeros((batch_size,1), dtype=torch.int)
                            rays = RaySamples(f, camera_indices=cam_indices).to(model.device)
                            field_outputs = vision_field.forward(rays)
                            rgb = field_outputs[FieldHeadNames.RGB]
                            density = field_outputs[FieldHeadNames.DENSITY]

                            rgbs.append(rgb.cpu())
                            densities.append(density.cpu())

                        rgb = torch.mean(torch.stack(rgbs, dim=0), dim=0)          # [batch_size, 3]
                        density = torch.mean(torch.stack(densities, dim=0), dim=0) # [batch_size, 1]

                    grid_sampler.update_feature_grid(batch_coords, rgb, density) 
                    progress.update(task, advance=batch_size)
                    torch.cuda.empty_cache()
            
        vision_field.spatial_distortion = spatial_distort

        if (visualize):
            grid_sampler.visualize_depth_grid(db_outs, show_poses=True)

        exit()

        # Save rgbsigma
        rgbsigma = grid_sampler.get_nerf_rpn_output()
        CONSOLE.print(f"[bold magenta]rgbsigma: {rgbsigma.size()}")
        rgbsigma = rgbsigma.cpu().numpy()
        min_pt = min_pt.cpu().numpy()
        max_pt = max_pt.cpu().numpy()
        CONSOLE.print(f"[bold green]Saving rgbsigma of size: {rgbsigma.shape}")
        np.savez_compressed(output_path, rgbsigma=rgbsigma, resolution=res,
                            bbox_min=min_pt, bbox_max=max_pt,
                            scale=0.3333, offset=0.0)

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
    Note: Only queries points within the scene bounding box.
    """
    nerf_model: Literal["splatfacto", "nerfacto", "zipnerf"] = "zipnerf"
    """Name of NeRF model utilized"""
    scene_name: str = "ai_001_001"
    """Name of the pre-trained scene"""
    dataset_type: Literal["hypersim"] = "hypersim"
    """Name of the dataset which will be used. TODO: Update for future datasets."""
    dataset_path: str = "hypersim"
    """The path to the scenes in instant-ngp data format."""
    transforms_filename: str = "transforms.json"
    """The name of the transforms file containing camera metadata, camera poses and bounding boxes."""
    ckpt_name: str = "step-000006999.ckpt"
    """Name of the snapshot/checkpoint"""
    max_res: int = 128
    """The maximum resolution of the extracted features."""
    crop_bbox: bool = True
    """Whether to crop the scene bounding box or not."""
    use_fixed_viewdirs: bool = False
    """Whether to use fixed view directions for sampling"""
    batch_size: int = 4096
    """Batch size for number of rays to compute"""
    visualize: bool = False
    """Whether to plot the extracted features in a 3D grid"""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        conf, pipeline, ckpt_path, _ = eval_setup(self.load_config)

        self.scene_name =  os.path.basename(conf.pipeline.datamanager.data)
        if (self.scene_name == "train"):
            # When instant-ngp dataset is used:
            p = os.path.normpath(conf.pipeline.datamanager.data)
            self.scene_name = p.split(os.sep)[-2]

        scene_dir = os.path.join(self.dataset_path, self.scene_name, 'train')
        if 'transforms.json' not in os.listdir(scene_dir):
            CONSOLE.print(f"[bold yellow]transforms.json not found for {self.scene_name}. Exiting.")
            sys.exit(1)
            
        json_path = os.path.join(scene_dir, self.transforms_filename)
        out_path = os.path.join(self.output_dir, f'{self.scene_name}.npz')

        if (self.nerf_model == "splatfacto"):
            assert isinstance(pipeline.model, SplatfactoModel)
            model: SplatfactoModel = pipeline.model
        elif (self.nerf_model == "nerfacto"):
            assert isinstance(pipeline.model, NerfactoModel)
            model: NerfactoModel = pipeline.model
        elif (self.nerf_model == "zipnerf"):
            from zipnerf_ns.zipnerf_model import ZipNerfModel 
            sys.path.append(r"C:\Users\OEM\nerf-gs-detect\nerfstudio\zipnerf-pytorch") 
            assert isinstance(pipeline.model, ZipNerfModel)
            model: ZipNerfModel = pipeline.model
        else:
            CONSOLE.print(f"[bold yellow]Invalid NeRF model: {self.nerf_model}. Exiting.")
            sys.exit(1)

        query_nerf_model(
            self.nerf_model, model, pipeline, json_path, out_path, 
            self.dataset_type, self.max_res, self.crop_bbox,
            self.use_fixed_viewdirs, self.batch_size, self.visualize
        )

        CONSOLE.print(f"[bold green]:white_check_mark: Saved features of extracted scene: {self.scene_name} ")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportNeRFRGBDensity, tyro.conf.subcommand(name="nerf-rgbd")], # Added
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

