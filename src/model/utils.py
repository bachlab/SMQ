from tqdm import tqdm
import torch

def get_num_actions(gt_dir, exclude = {"background"}):
    """
    Count unique action labels in groundTruth/*.txt 
    (excluding background tokens).
    """
    if not gt_dir.is_dir():
        return 0
    uniq = set()
    for p in gt_dir.glob("*.txt"):
        try:
            with p.open("r") as f:
                for line in f:
                    lab = line.strip()
                    if lab and lab not in exclude:
                        uniq.add(lab)
        except Exception:
            pass
    return len(uniq)


def print_run_summary(
    dataset: str,
    num_features: int,
    num_joints: int,
    num_person: int,
    num_actions: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patch_size: int
) -> None:
    """
    Print run summary before training. 
    """

    info = {
        "Dataset": dataset,
        "Features": num_features,
        "Num Joints": num_joints,
        "Num Skeleton" : num_person,
        "Num Actions": num_actions,
        "Patch Size": patch_size,
        "Epochs": epochs,
        "Batch": batch_size,
        "LR": learning_rate,
    }

    summary = "  ".join(f"{k}: {v}" for k, v in info.items())
    tqdm.write(f"[Run Config] {summary}")

def distance_joints(data):
    """
    Computes pairwise distances between all joints in the current frame.

    Parameters:
    data (torch.Tensor): The input tensor with shape (batch_size, num_features, seq_len, num_joints, num_skeletons).
    
    Returns:
    torch.Tensor: Distance tensor with shape (batch_size, num_skeletons, seq_len, num_joints, num_joints)
    """
    
    # Reshape and permute to organize dimensions
    # From (batch_size, num_features, seq_len, num_joints, num_skeletons)
    # To   (batch_size, num_skeletons, seq_len, num_joints, num_features)
    data = data.permute(0, 4, 2, 3, 1)
    
    # Compute pairwise distances between all joints
    distance = torch.cdist(data, data, p=2)  # Shape: (batch_size, num_skeletons, seq_len, num_joints, num_joints)
    
    return distance


def process_mask(mask_joint, batch_size, num_person, num_joints, seq_length, num_features_per_joint):
    """
    Build VQ mask for concatenated (person, joint) latent tokens.

    Args:
        mask_joint: (N*M*V, C, T) mask in joint packed layout (same as encoder input)
        batch_size: N
        num_person: M
        num_joints: V
        seq_length: T
        num_features_per_joint: latent_dim (Z)

    Returns:
        vq_mask: (N, T, V*M*latent_dim)
    """
    # take the first channel of the mask
    m = mask_joint[:, 0, :]                                  # (N*M*V, T)
    m = m.view(batch_size, num_person, num_joints, seq_length)  # (N, M, V, T)
    m = m.permute(0, 3, 2, 1).contiguous()                   # (N, T, V, M)

    # expand per-joint validity to per-joint latent_dim features
    m = m.unsqueeze(-1).repeat(1, 1, 1, 1, num_features_per_joint)  # (N, T, V, M, Z)

    # flatten to match token dim (V*M*Z)
    return m.view(batch_size, seq_length, num_joints * num_person * num_features_per_joint)
