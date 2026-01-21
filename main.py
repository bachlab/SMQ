from pathlib import Path
import argparse
import random

import torch

from src.model.utils import get_num_actions, print_run_summary
from batch_gen import BatchGenerator
from model import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# -------------------------------
# Dataset Defaults
# -------------------------------

def get_dataset_defaults(name: str):
    if name == "hugadb":
        return dict(batch_size=8,  num_features=6, num_joints=6,  num_person=1,
                    patch_size=60)
    if name == "lara":
        return dict(batch_size=8,  num_features=6, num_joints=22, num_person=1,
                    patch_size=50)
    if name in {"babel1", "babel2", "babel3"}:
        return dict(batch_size=32, num_features=3, num_joints=25, num_person=1,
                    patch_size=30)
    raise ValueError(f"Unknown dataset: {name}")

# -------------------------------
# CLI
# -------------------------------

parser = argparse.ArgumentParser(description='Train and eval pipeline for SMQ')

# Core action/dataset
parser.add_argument("--action", choices=["train", "eval"], default="train", help="Whether to train or eval.")
parser.add_argument("--dataset", required=True, choices=["hugadb", "lara", "babel1", "babel2", "babel3"])
parser.add_argument("--ckpt", type=Path, default=None, help="Path to a .model checkpoint to use for eval. " "If not set, uses models/<dataset>/epoch-<epoch>.model.")

# Training & model parameters
parser.add_argument("--epoch", type=int, default=30, help="Number of epochs.")
parser.add_argument("--batch_size", type=int, help="Batch size (overrides dataset default).")
parser.add_argument("--num_f_maps", type=int, default=128, help="Number of feature maps.")
parser.add_argument("--num_layers", type=int, default=3, help="Number of TCN dilated residual layers per stage.")
parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension per joint.")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
parser.add_argument("--sample_rate", type=int, default=1, help="Temporal subsampling rate (use every k-th frame).")

# VQ parameters
parser.add_argument("--patch_size", type=int, help="Patch size for quantization (overrides dataset default).")
parser.add_argument("--num_actions", type=int, help="Number of actions (overrides dataset default).")
parser.add_argument("--kmeans", action="store_true", help="Use K-Means for codebook initialization.")
parser.add_argument("--kmeans_metric", type=str, choices=["euclidean", "dtw"], default="euclidean", help="Metric for K-Means init.")
parser.add_argument("--sampling_quantile", type=float, default=0.5, help="Quantile used for selecting candidate patches when replacing dead codes.")
parser.add_argument("--replacement_strategy", type=str, choices=["representative", "exploratory"], default="representative", help="Dead-code replacement strategy: ""'representative' picks well-covered patches; ""'exploratory' picks poorly-covered (farther) patches.")
parser.add_argument("--decay", type=float, default=0.5, help="Decay weight.")

# Loss parameters
parser.add_argument("--mse_loss_weight", type=float, default=0.001, help="Reconstruction loss weight.")
parser.add_argument("--commit_weight", type=float, default=1.0, help="Commitment loss weight.")
parser.add_argument("--joint_distance_recons", action=argparse.BooleanOptionalAction, default=True, help="Use joint-distance reconstruction loss (default: True).")
parser.add_argument("--vis", action="store_true", help="Enable segmentation visualization during eval.")

# Paths
parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root for datasets.")
parser.add_argument("--models_root", type=Path, default=Path("./models"), help="Root for model checkpoints.")
parser.add_argument("--vis_root", type=Path, default=Path("./vis"), help="Root for visualizations.")

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Paths
    dataset_root = args.data_root / args.dataset
    features_path = dataset_root / "features"
    gt_path = dataset_root / "groundTruth"
    mapping_file = dataset_root / "mapping" /"mapping.txt"
    model_dir = args.models_root / args.dataset
    plot_dir = args.vis_root / args.dataset

    # Dataset defaults + simple overrides
    cfg          = get_dataset_defaults(args.dataset)
    batch_size   = args.batch_size  if args.batch_size  is not None else cfg["batch_size"]
    patch_size   = args.patch_size  if args.patch_size  is not None else cfg["patch_size"]
    num_features = cfg["num_features"]
    num_joints   = cfg["num_joints"]
    num_person   = cfg["num_person"]
    num_actions_calc = get_num_actions(gt_path)
    num_actions = args.num_actions if args.num_actions is not None else num_actions_calc
    
    # Ensure output dirs exist
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.vis :
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Build trainer
    trainer = Trainer(
        in_channels = num_features,
        filters = args.num_f_maps,
        num_layers = args.num_layers,
        latent_dim = args.latent_dim,
        num_actions = num_actions,
        num_joints = num_joints,
        num_person = num_person,
        patch_size = patch_size,
        kmeans=args.kmeans,
        kmeans_metric=args.kmeans_metric,
        sampling_quantile=args.sampling_quantile,
        replacement_strategy=args.replacement_strategy,
        decay=args.decay,
    )

    # Execute action
    if args.action == "train":

        batch_gen = BatchGenerator(features_path=features_path, 
                                   sample_rate=args.sample_rate, 
                                   num_features=num_features, 
                                   num_joints=num_joints, 
                                   num_person=num_person)
        
        # Read features
        batch_gen.read_data() 

        # Print run summary
        print_run_summary(
        dataset=args.dataset,
        num_features=cfg.get("num_features"),
        num_joints=cfg.get("num_joints"),
        num_person=cfg.get("num_person"),
        num_actions=num_actions,
        epochs=args.epoch,
        batch_size=batch_size,
        learning_rate=args.lr,
        patch_size=patch_size)

        trainer.train(
            save_dir=model_dir,
            batch_gen=batch_gen,
            num_epochs=args.epoch,
            batch_size=batch_size,
            learning_rate=args.lr,
            commit_weight=args.commit_weight,
            mse_loss_weight=args.mse_loss_weight,
            device=device,
            joint_distance_recons = args.joint_distance_recons
        )

    elif args.action == "eval":
        
        # Use models/pretrained model or models/<dataset>
        if args.ckpt is not None:
            ckpt_path = args.ckpt
        else:
            ckpt_path = model_dir / f"epoch-{args.epoch}.model"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        trainer.eval(
            model_path=ckpt_path,
            features_path=features_path,
            gt_path=gt_path,
            mapping_file=mapping_file,
            epoch=args.epoch,
            vis=args.vis,
            plot_dir=plot_dir,
            device=device
        )
    
    else:
        raise ValueError(f"Wrong action! Available choices : [train, eval]")
