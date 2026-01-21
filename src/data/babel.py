import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm

# Preset action dictionaries
BABEL_PRESETS = {
    "babel1": {
        0: "walk",
        1: "stand",
        2: "turn",
        3: "jump",
        4: "none",
    },
    "babel2": {
        0: "sit",
        1: "run",
        2: "stand_up",
        3: "kick",
        4: "none",
    },
    "babel3": {
        0: "jog",
        1: "wave",
        2: "dance",
        3: "gesture",
        4: "none",
    },
}


def create_mapping_file(actions_dict, mapping_folder):
    os.makedirs(mapping_folder, exist_ok=True)
    mapping_path = os.path.join(mapping_folder, "mapping.txt")
    # Avoid trailing newline at EOF
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{k} {v}" for k, v in sorted(actions_dict.items())))
    print(f"Mapping file created at {mapping_path}")


def save_sequence(sid, X, L, actions_dict, features_dir, groundtruth_dir):
    
    # Convert meters to millimeters
    X = X * 1000.0

    # Save features
    feat_path = os.path.join(features_dir, f"{sid}.npy")
    np.save(feat_path, X)

    # Save labels as action strings, no trailing newline
    gt_path = os.path.join(groundtruth_dir, f"{sid}.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(actions_dict.get(int(lbl), "Unknown") for lbl in L))


def process_dataset(pickle_files, actions_dict, output_folder, preset, none_label=4, none_threshold=50):
    # Create a subfolder named after the preset
    subset_folder = os.path.join(output_folder, preset)
    features_dir = os.path.join(subset_folder, "features")
    groundtruth_dir = os.path.join(subset_folder, "groundTruth")
    mapping_dir = os.path.join(subset_folder, "mapping")

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    create_mapping_file(actions_dict, mapping_dir)

    excluded = []
    kept = 0

    for pkl in pickle_files:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        sids = data["sid"]
        Xs = data["X"]
        Ls = data["L"]

        for i in tqdm(range(len(sids)), desc=f"Processing {os.path.basename(pkl)} ({preset})", ncols=100):
            sid = sids[i]
            X = Xs[i]
            L = Ls[i]

            # Compute 'none' percentage
            none_count = (L == none_label).sum()
            none_pct = (float(none_count) / len(L)) * 100.0 if len(L) > 0 else 100.0

            if none_pct > none_threshold:
                excluded.append(sid)
                continue

            save_sequence(sid, X, L, actions_dict, features_dir, groundtruth_dir)
            kept += 1

    print(f"Number of excluded files: {len(excluded)}")
    print(f"Number of remaining files: {kept}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BABEL subsets for SMQ")
    parser.add_argument(
        "-i", "--inputs",
        nargs="+",
        required=True,
        help="Paths to one or more BABEL pickle files (e.g., train_split*.pkl val_split*.pkl)",
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default='data/babel1',
        help="Root output folder (subfolder named after preset will be created)",
    )
    parser.add_argument(
        "-p", "--preset",
        choices=sorted(BABEL_PRESETS.keys()),
        required=True,
        help="Action mapping preset: babel1 | babel2 | babel3",
    )
    parser.add_argument(
        "-n", "--none-label",
        type=int,
        default=4,
        help="Numeric label id corresponding to 'none' in labels (default: 4)",
    )
    parser.add_argument(
        "-t", "--none-threshold",
        type=float,
        default=50.0,
        help="Percentage threshold for 'none' to exclude a sequence (default: 50)",
    )

    args = parser.parse_args()

    actions_dict = BABEL_PRESETS[args.preset]
    process_dataset(
        pickle_files=args.inputs,
        actions_dict=actions_dict,
        output_folder=args.output_folder,
        preset=args.preset,
        none_label=args.none_label,
        none_threshold=args.none_threshold
    )


if __name__ == "__main__":
    main()
