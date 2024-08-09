"""
Automated Hypersim training for Nerfacto, TensoRF, ZipNeRF 
python train_all_hyp_nerf.py --db_split_dir .../hypersim_split.npz --nerf_model tensorf
"""

import numpy as np
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Configure ImVoxelNet, NeRF-RPN or NeRF-Det')

    parser.add_argument('--db_split_dir',required=True, help='Path to the hypersim split .npz file')
    parser.add_argument('--nerf_model', type=str, required=True, choices=['nerfacto', 'tensorf', 'zipnerf'],
                        help='Which NeRF model to train')
    args = parser.parse_args()
    return args

args = parse_args()
data = np.load(args.db_split_dir)
train_scenes = data['train_scenes'][:1]
nerf_model = args.nerf_model

base_command = "ns-train {} --experiment-name nerf_data --max-num-iterations 10000 --steps-per-save 1000 --viewer.quit-on-train-completion True --output-dir {} instant-ngp-data --data {}"

dataset_dir = r'E:\NeRF_datasets\hypersim_ngp_format'

for scene in train_scenes:
    output_dir = f"{dataset_dir}\\{scene}\\train"
    db_dir = f"{dataset_dir}\\{scene}\\train"
    
    command = base_command.format(nerf_model, output_dir, db_dir)
    process = subprocess.Popen(command, shell=True)
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print(f"Training interrupted for scene {scene}")

    process.terminate()
    print(f"Finished training for scene {scene}")