"""
This script converts raw HuGaDB tab-separated files into:
• NumPy feature tensors (.npy)
• Per-frame ground-truth labels (.txt) using a mapping file

Usage:
python src/data/hugadb.py \
--input-folder "/path/to/raw/HuGaDB/" \
--output-folder "/path/to/output/"

Outputs:
mapping/mapping.txt
features/<basename>.npy
groundTruth/<basename>.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Action dictionary based on the provided mappings
actions_dict = {
    1: 'walking',
    2: 'running',
    3: 'going_up',
    4: 'going_down',
    5: 'sitting',
    6: 'sitting_down',
    7: 'standing_up',
    8: 'standing',
    9: 'bicycling',
    10: 'up_by_elevator',
    11: 'down_by_elevator',
    12: 'sitting_in_car'
}

def create_mapping_file(actions_dict, mapping_folder):
    # Create mapping folder and write mapping.txt
    os.makedirs(mapping_folder, exist_ok=True)
    mapping_path = os.path.join(mapping_folder, 'mapping.txt')
    with open(mapping_path, 'w') as file:
        file.write('\n'.join(f"{key} {value}" for key, value in sorted(actions_dict.items())))
    print(f"Mapping file created at {mapping_path}")


def process_file(input_file_path, features_dir, ground_truth_dir):
    # Ensure output directories exist
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)

    # Read the input file
    df = pd.read_csv(input_file_path, sep='\t', skiprows=3)

    # Features and actions
    actions = df.iloc[:, -1].values  # Last column for actions
    features = df.iloc[:, :-3].values  # Drop the last 3 columns - EMG + action

    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file_path))[0]

    # Reshape features into skeleton form
    seq_len, num_features = features.shape
    features = features.reshape(seq_len, 1, 6, 6)  # 1 -> num skeleton/ 6-> num joints/ 6 -> num features
    features = features.transpose(3, 0, 2, 1)

    # Save features as npy file
    feature_file_path = os.path.join(features_dir, f"{base_filename}.npy")
    np.save(feature_file_path, features)

    # Convert actions to their string representation
    action_strings = [actions_dict.get(int(action), 'unknown') for action in actions]

    # Save actions as a text file
    ground_truth_file_path = os.path.join(ground_truth_dir, f"{base_filename}.txt")
    with open(ground_truth_file_path, 'w') as f:
        f.write('\n'.join(action_strings))


def process_folder(input_folder, output_folder):
    # Prepare subfolders inside the output folder
    features_folder = os.path.join(output_folder, 'features')
    ground_truth_folder = os.path.join(output_folder, 'groundTruth')
    mapping_folder = os.path.join(output_folder, 'mapping')

    # Create mapping file once
    create_mapping_file(actions_dict, mapping_folder)

    # Gather all .txt files
    txt_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.txt')]

    # Use tqdm progress bar
    for file_name in tqdm(txt_files, desc='Processing files', ncols=100):
        file_path = os.path.join(input_folder, file_name)
        process_file(file_path, features_folder, ground_truth_folder)


def main():
    parser = argparse.ArgumentParser(description='Prepare HuGaDB data for SMQ')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Path to the input folder containing raw HuGaDB .txt files')
    parser.add_argument('-o', '--output_folder', type=str, default='data/hugadb/', help='Path to the output folder where subfolders will be created')
    args = parser.parse_args()

    # Always write into <output_root>/<dataset_name>/
    output_dataset_dir = os.path.join(args.output_folder, "hugadb")
    process_folder(args.input_folder, output_dataset_dir)

if __name__ == '__main__':
    main()




