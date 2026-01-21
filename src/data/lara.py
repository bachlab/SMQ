import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# LARa actions mapping
actions_dict = {
    1: "Standing",
    2: "Walking",
    3: "Cart",
    4: "Handling(upwards)",
    5: "Handling(centred)",
    6: "Handling(downwards)",
    7: "Synchronization",
    8: "None"}


def create_mapping_file(actions_dict, mapping_folder):
    os.makedirs(mapping_folder, exist_ok=True)
    mapping_path = os.path.join(mapping_folder, "mapping.txt")
    # Avoid trailing newline
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{k} {v}" for k, v in sorted(actions_dict.items())))
    print(f"Mapping file created at {mapping_path}")


def normalize_to_hip(features_np, hip_index):
    # features_np shape: (T, 1, 22, 6); last 3 features are pose coord.

    static_features = features_np[:, :, :, :3]
    xyz_features = features_np[:, :, :, 3:]

    # Normalize by subtracting the hip joint's coordinates at each frame
    hip_coords = xyz_features[:, :, hip_index, :]  # (T, 1, 3)
    xyz_features = xyz_features - hip_coords[:, :, np.newaxis, :]

    # Reassemble the array
    normalized = np.concatenate((static_features, xyz_features), axis=3)
    return normalized


def process_features(input_file_path, output_file_path, sample_rate, hip_index):
    # Read CSV, skip 4 header rows and drop first two columns
    df = pd.read_csv(input_file_path, skiprows=4)
    df = df.drop(df.columns[[0, 1]], axis=1)

    # Downsample by factor of sample rate (default = 4)
    df = df.iloc[::sample_rate]
    features_np = df.to_numpy()

    # Reshape to (T, 1, 22, 6)
    seq_len, _ = features_np.shape
    features_np = features_np.reshape(seq_len, 1, 22, 6)

    # Normalize to hip
    features_np = normalize_to_hip(features_np, hip_index)

    # Transpose to (6, T, 22, 1)
    features_np = features_np.transpose(3, 0, 2, 1)

    # Save
    np.save(output_file_path, features_np)


def process_labels(file_path, output_file_path, sample_rate):
    try:
        df = pd.read_csv(file_path)
        # first 8 columns correspond to action one-hot; downsample by 4
        sliced = df.iloc[:, :8].iloc[::sample_rate]
        labels = []
        for _, row in sliced.iterrows():
            cols_true = row[row == 1]
            if len(cols_true) == 0:
                labels.append("None")  # fallback if no action detected in the row
            else:
                labels.append(cols_true.index[0])
        # Avoid trailing newline
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file '{file_path}'.")


def find_feature_csv_files(folder_path):
    feature_csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv") and not file.endswith("labels.csv"):
                feature_csv_files.append(os.path.join(root, file))
    return sorted(feature_csv_files)


def find_label_csv_files(folder_path):
    label_csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("labels.csv"):
                label_csv_files.append(os.path.join(root, file))
    return sorted(label_csv_files)


def process_all(input_folder, output_folder, sample_rate, hip_index):
    # Prepare subfolders under output
    features_folder = os.path.join(output_folder, "features")
    labels_folder = os.path.join(output_folder, "groundTruth")
    mapping_folder = os.path.join(output_folder, "mapping")

    os.makedirs(features_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    create_mapping_file(actions_dict, mapping_folder)

    feature_files = find_feature_csv_files(input_folder)
    label_files = find_label_csv_files(input_folder)

    # Features
    for fpath in tqdm(feature_files, desc="Features", ncols=100):
        fname = os.path.basename(fpath)[:-3] + "npy"
        out_path = os.path.join(features_folder, fname)
        process_features(fpath, out_path, sample_rate, hip_index)

    # Labels
    for lpath in tqdm(label_files, desc="Labels   ", ncols=100):
        base = os.path.basename(lpath)[:-11]  # strip 'labels.csv'
        out_txt = os.path.join(labels_folder, base + ".txt")
        process_labels(lpath, out_txt, sample_rate)

    print(f"Processed {len(feature_files)} feature files and {len(label_files)} label files.")


def main():
    parser = argparse.ArgumentParser(description="Prepare LARa dataset for SMQ")
    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        required=True,
        help="Path to the input folder containing raw LARa CSV files",
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="data/lara/",
        help="Output folder (features/, groundTruth/, mapping/ will be created)",
    )
    parser.add_argument(
        "-sr", "--sample-rate",
        type=int,
        default=4,
        help="Downsampling rate for frames (default: 4)",
    )
    parser.add_argument(
        "-hip", "--hip-index",
        type=int,
        default=21,
        help="Index of the hip joint used for normalization (default: 21)",
    )

    args = parser.parse_args()
    # Always write into <output_root>/<dataset_name>/
    output_dataset_dir = os.path.join(args.output_folder, "lara")
    process_all(args.input_folder, output_dataset_dir, args.sample_rate, args.hip_index)


if __name__ == "__main__":
    main()
