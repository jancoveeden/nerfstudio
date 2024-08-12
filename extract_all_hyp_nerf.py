"""
Automated Hypersim extract for Nerfacto, TensoRF, ZipNeRF 

Example of usage:
python extract_all_hyp_nerf.py 
--db_split_dir ...\hypersim_split.npz
--nerf_model nerfacto
--max_res 200
"""

import numpy as np
import subprocess
import argparse
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Automated Hypersim extract for Nerfacto, TensoRF, ZipNeRF')

    parser.add_argument('--db_split_dir',required=True, help='Path to the hypersim split .npz file')
    parser.add_argument('--nerf_model', type=str, required=True, choices=['nerfacto', 'tensorf', 'zipnerf'],
                        help='Which NeRF model to train')
    parser.add_argument('--max_res', default=128, type=int,
                        help='The maximum resolution of the extracted features.')
    args = parser.parse_args()
    return args

args = parse_args()
nerf_model = args.nerf_model
max_res = args.max_res
batch_size = 16384
dataset_name = 'hypersim'

if (nerf_model == 'nerfacto'):
    output_dir = r'E:\NeRF_datasets\nerf_rpn\hyp_nerfacto_rpn_data\features'
elif (nerf_model == 'tensorf'):
    output_dir = r'E:\NeRF_datasets\nerf_rpn\hyp_tensorf_rpn_data\features'
elif (nerf_model == 'zipnerf'):
    output_dir = r'E:\NeRF_datasets\nerf_rpn\hyp_zipnerf_rpn_data\features'
else:
    print(f'Invalid nerf_model: {nerf_model}')
    exit()

base_cmd = "ns-export nerf-rgbd --load-config {} --output-dir {} --dataset-path {} --nerf_model {} --dataset_type {} --batch_size {} --max_res {}"

dataset_dir = r'E:\NeRF_datasets\hypersim_ngp_format'

data = np.load(args.db_split_dir)
scenes_names = np.concatenate((data['train_scenes'], data['test_scenes'], data['val_scenes']))

for scene in scenes_names:
    scene_directory = os.path.join(dataset_dir, scene, 'train', 'nerf_data', nerf_model)
    date_time_folders = glob.glob(os.path.join(scene_directory, '*'))

    if date_time_folders:
        date_time_folder = date_time_folders[0]
        config_file_path = os.path.join(date_time_folder, 'config.yml')

        if not os.path.exists(config_file_path):
            print(f'Config file not found in {date_time_folder}')
            continue
    else:
        print(f'No trained NeRF model found at: {scene_directory}')
        continue
    
    command = base_cmd.format(config_file_path, output_dir, dataset_dir, 
                              nerf_model, dataset_name, batch_size, max_res)
    process = subprocess.Popen(command, shell=True)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print(f"Exporting interrupted for scene: {scene}")

    process.terminate()
    print(f"Finished exporting for scene: {scene}")