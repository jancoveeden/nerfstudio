"""
Automated Hypersim training for Nerfacto, TensoRF, ZipNeRF 
python train_all_hyp_nerf.py --db_split_dir .../hypersim_split.npz --nerf_model tensorf
"""

import numpy as np
import subprocess
import argparse
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Automated Hypersim training for Nerfacto, TensoRF, ZipNeRF')

    parser.add_argument('--db_split_dir',required=True, 
                        help='Path to the hypersim split .npz file')
    parser.add_argument('--nerf_model', type=str, required=True, choices=['nerfacto', 'tensorf', 'zipnerf'],
                        help='Which NeRF model to train')
    parser.add_argument('--check_scenes', action='store_true',
                        help='Checks and counts the number of scenes trained and exits.')
    args = parser.parse_args()
    return args

args = parse_args()
data = np.load(args.db_split_dir)
scenes_names = np.concatenate((data['train_scenes'], data['test_scenes'], data['val_scenes']))
nerf_model = args.nerf_model

base_command = "ns-train {} --experiment-name nerf_data --max-num-iterations 10000 --steps-per-save 2000 --viewer.quit-on-train-completion True --output-dir {} instant-ngp-data --data {}"

dataset_dir = r'E:\NeRF_datasets\hypersim_ngp_format'

if (args.check_scenes):
    count = 0
    for scene in scenes_names:
        scene_directory = os.path.join(dataset_dir, scene, 'train', 'nerf_data', nerf_model)
        if os.path.exists(scene_directory):
            date_time_folders = glob.glob(os.path.join(scene_directory, '*'))

            if len(date_time_folders) > 1:
                print(f"Found {len(date_time_folders)} configurations for scene: {scene}")

            for time_folder in date_time_folders:
                models_path = os.path.join(time_folder, 'nerfstudio_models')

                if os.path.exists(models_path):
                    checkpoint_p = glob.glob(os.path.join(models_path, '*'))
                    checkpoint_p = checkpoint_p[0]

                    if os.path.exists(checkpoint_p):
                        iterations = int(checkpoint_p[-14:-5])
                        if (iterations > 6999):
                            count += 1
                        else:
                            print(f"Scene: {scene} found, but only trained with {iterations} iterations")
                else:
                    print(f"Scene: {scene} has no checkpoint, but found config")

    print(f"Scenes trained above 7k iterations: {count}")
    print(f"Number of scenes left to train: {len(scenes_names) - count}")
    exit()

for c, scene in enumerate(scenes_names):
    scene_directory = os.path.join(dataset_dir, scene, 'train', 'nerf_data', nerf_model)

    if os.path.exists(scene_directory):
        print(f"Found trained {nerf_model} for scene: {scene}. Skipping...")
        continue

    output_dir = f"{dataset_dir}\\{scene}\\train"
    db_dir = f"{dataset_dir}\\{scene}\\train"
    
    command = base_command.format(nerf_model, output_dir, db_dir)
    process = subprocess.Popen(command, shell=True)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print(f"Training interrupted for scene: {scene}")

    process.terminate()
    print(f"Finished training for scene: {scene}. Scenes trained so far: {c+1}")