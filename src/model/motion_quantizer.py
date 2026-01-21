# =============================================================================
# vq.py — SMQ module - patch based quantization
# Adapted from : https://github.com/lucidrains/vector-quantize-pytorch
# =============================================================================

import torch 
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

from tslearn.clustering import TimeSeriesKMeans


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    """
    Apply Laplace smoothing to avoid zero counts in categorical statistics.
    """
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def kmeans_time_series(samples, num_clusters, num_iters=10, metric='euclidean', random_state=42):
    """
    Initialize codebook using time-series K-Means clustering.
    """
    samples_np = samples.cpu().detach().numpy()
    
    kmeans = TimeSeriesKMeans(
        n_clusters=num_clusters,
        max_iter=num_iters,
        metric=metric,
        random_state=random_state
    )
    kmeans.fit(samples_np)
    
    means = torch.tensor(kmeans.cluster_centers_, device=samples.device)
    labels = torch.tensor(kmeans.labels_, device=samples.device)

    # Count how many samples were assigned to each cluster
    cluster_sizes = torch.bincount(labels, minlength=num_clusters).float().view(-1, 1, 1)

    return means, cluster_sizes


def euclidean_dist(ts1, ts2):
    """
    Compute pairwise Euclidean distances between two sets of temporal patches.
    """
    ts1 = ts1.unsqueeze(1)  # Shape: (N1, 1, window, embedding_dim)
    ts2 = ts2.unsqueeze(0)  # Shape: (1, N2, window, embedding_dim)

    # Frame-wise squared differences
    diff = ts1 - ts2
    squared_diff = diff ** 2
    
    # Euclidean distance per frame
    sum_squared_diff = torch.sum(squared_diff, dim=-1)
    distances = torch.sqrt(sum_squared_diff)
    
    # Sum over time window to get patch-level distance
    distances = torch.sum(distances, dim=-1)

    return distances


class SkeletonMotionQuantizer(nn.Module):
    """
    Skeleton Motion Quantization (SMQ)

    Quantizes patches into discrete codebook entries,
    using EMA update and optional time series K-Means initialization. 
    Includes dead-code replacement for stability.
    """
    def __init__(self, num_embeddings, embedding_dim, window, commitment_cost, 
                 decay=0.8, eps=1e-5, threshold_ema_dead_code=10, sampling_quantile=0.5,
                replacement_strategy = "representative", kmeans=False, kmeans_metric='euclidean'):

        super(SkeletonMotionQuantizer, self).__init__()

        # Codebook configuration
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._window = window
        self.kmeans_metric = kmeans_metric
        self.kmeans = kmeans

        # Codebook: (num_embeddings, window, embedding_dim)
        self._embedding = nn.Parameter(torch.zeros(num_embeddings, window, embedding_dim))
        
        # VQ loss weight
        self._commitment_cost = commitment_cost

        # EMA and dead-code handling parameters
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sampling_quantile = sampling_quantile
        self.replacement_strategy = replacement_strategy

        # EMA buffers
        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings, 1, 1))
        self.register_buffer('embed_avg', self._embedding.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """
        Initialize the codebook once, using either K-Means or random weights.
        """
        if self.initted:
            return

        if self.kmeans:
            # Initialize codebook using K-Means
            embed, cluster_size = kmeans_time_series(data, self._num_embeddings, num_iters=20,
                                                     metric=self.kmeans_metric)
            embed_sum = embed * cluster_size
            self._embedding.data.copy_(embed)
            self.embed_avg.data.copy_(embed_sum)
            self.cluster_size.data.copy_(cluster_size)
        else:
            # Uniform initialization
            embed = torch.empty_like(self._embedding)
            nn.init.kaiming_uniform_(embed)
            self._embedding.data.copy_(embed)
            self.embed_avg.data.copy_(embed)
            self.cluster_size.data.fill_(1)

        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask, random_generator=None):
        """
        Reinitialize dead codes using patches sampled from the current batch.

        Args:
            batch_samples: (N, W, D) patches from current batch (valid patches)
            batch_mask: (K,) bool mask indicating which codes are dead
            random_generator: torch.Generator or None for deterministic sampling
        """
        dead_code_indices = batch_mask.nonzero(as_tuple=False).flatten()
        num_dead_codes = dead_code_indices.numel()
        if num_dead_codes == 0:
            return

        # Total patches we want to sample
        total_samples_needed = num_dead_codes * self.threshold_ema_dead_code

        # Compute distances between batch_samples and embeddings
        distances = -euclidean_dist(batch_samples, self._embedding)
        min_distances, _ = distances.max(dim=1)

        # Compute quantile
        quantile_value = torch.quantile(min_distances, self.sampling_quantile)

        # Choose candidate patches based on replacement strategy to reinitialize dead codes
        if self.replacement_strategy == "representative":
            candidate_indices = (min_distances >= quantile_value).nonzero(as_tuple=False).flatten() 
        
        elif self.replacement_strategy == "exploratory":
            candidate_indices = (min_distances <= quantile_value).nonzero(as_tuple=False).flatten() 
        else:
            raise ValueError(f"Unknown replacement_strategy: {self.replacement_strategy}")

        # If no indices as candidate, just allow any patch as a candidate
        if candidate_indices.numel() == 0:
            candidate_indices = torch.arange(batch_samples.shape[0], device=batch_samples.device)

        # Select indices to sample
        if len(candidate_indices) < total_samples_needed:
            selected_indices = candidate_indices
        
        else:
            permuted_indices = torch.randperm(len(candidate_indices), generator=random_generator)
            selected_indices = candidate_indices[permuted_indices[:total_samples_needed]]

        sampled = batch_samples[selected_indices]  # (total_needed, W, D) or fewer if not enough

        # If there is not enough patches for all dead codes repeat-to-fill
        if sampled.shape[0] < total_samples_needed:    
            print(
                f"[VQ] Warning: Only {sampled.shape[0]} candidate patches available, "
                f"but {total_samples_needed} are needed to replace {num_dead_codes} dead codes. "
                "Repeating samples to fill the requirement.")
            # Repeat cyclically to fill (only happens when candidate pool is tiny)
            repeat_factor = (total_samples_needed + sampled.shape[0] - 1) // sampled.shape[0]
            sampled = sampled.repeat(repeat_factor, 1, 1)[:total_samples_needed]
        
        # Assign sampled patches to each dead code and update buffers
        sampled = sampled.reshape(num_dead_codes, self.threshold_ema_dead_code, *batch_samples.shape[1:])
        sampled_means = sampled.mean(dim=1)

        # Update embeddings and EMA buffers for the dead codes
        for i, code_idx in enumerate(dead_code_indices):
            self._embedding.data[code_idx] = sampled_means[i]
            self.cluster_size.data[code_idx] = self.threshold_ema_dead_code
            self.embed_avg.data[code_idx] = sampled_means[i] * self.threshold_ema_dead_code

    def expire_codes_(self, batch_samples, random_generator=None):
        """
        Detect dead codes from EMA cluster_size and replace them using batch_samples.
        """
        if self.threshold_ema_dead_code == 0:
            return

        # cluster_size: (K,1,1) -> (K,)
        cluster_size_flat = self.cluster_size.squeeze()
        expired_codes = cluster_size_flat < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        self.replace(batch_samples, expired_codes, random_generator=random_generator)

    def forward(self, x, mask):
        """
        Args:
            x:    (B, T, D) float
            mask: (B, T, D) float/bool, 1=valid, 0=pad
        Returns:
            quantize:        (B, T, D)
            encoding_indices:(B, T)      code id per frame (patch id repeated)
            loss:            scalar      commitment loss
            distances:       (B, T, K)   (negative) distances per frame to each code
        """
        B, T, D = x.shape
        W = self._window
        K = self._num_embeddings

        # Pad time dimension if necessary
        remainder = T % W
        padding_needed = (W - remainder) if remainder != 0 else 0

        x_pad = F.pad(x,    (0, 0, 0, padding_needed), mode="constant", value=0)
        mask_pad = F.pad(mask, (0, 0, 0, padding_needed), mode="constant", value=0)

        _, T_pad, _ = x_pad.shape
        P = T_pad // W  # patches per sequence

        # Patchify: (B, T_pad, D) -> (B*P, W, D)
        x_patches = x_pad.reshape(B * P, W, D)
        mask_patches = mask_pad.reshape(B * P, W, D)

        # valid patch = contains at least one valid element
        valid_patch_mask = mask_patches.sum(dim=(1, 2)) > 0          # (B*P,)
        valid_patches = x_patches[valid_patch_mask]                  # (N_valid, W, D)

        # Codebook init
        self.init_embed_(valid_patches)

        # Assign codes to patches
        distances_valid = -euclidean_dist(valid_patches, self._embedding)  # (N_valid, K)
        encoding_indices_valid = torch.argmax(distances_valid, dim=1)      # (N_valid,)
        encoding_onehot_valid = F.one_hot(encoding_indices_valid, K).float()  # (N_valid, K)

        # EMA codebook update
        if self.training:
            cluster_size = encoding_onehot_valid.sum(dim=0).unsqueeze(1).unsqueeze(-1)  # (K,1,1)
            self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

            embed_sum = einsum('ijk,il->ljk', valid_patches, encoding_onehot_valid)     # (K,W,D)
            self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, K) * self.cluster_size.sum(dim=-1, keepdim=True)
            embed_normalized = self.embed_avg / cluster_size
            self._embedding.data.copy_(embed_normalized)

            random_generator = torch.Generator()
            random_generator.manual_seed(42)
            self.expire_codes_(valid_patches, random_generator=random_generator)

        # Quantization
        quantize_valid = torch.sum(
            encoding_onehot_valid.unsqueeze(-1).unsqueeze(-1) * self._embedding,
            dim=1
        )  # (N_valid, W, D)

        # Put quantized patches back into the full patch tensor (invalid -> 0)
        quantized_patches = torch.zeros_like(x_patches)         # (B*P, W, D)
        quantized_patches[valid_patch_mask] = quantize_valid
        quantize_pad = quantized_patches.reshape(B, T_pad, D)   # (B, T_pad, D)

        # Crop back to original length T
        end = -padding_needed if padding_needed > 0 else None
        x_crop = x_pad[:, :end, :]
        quantize = quantize_pad[:, :end, :]

        # Commit loss
        loss_mask = mask[:, :x_crop.shape[1], :]
        diff = quantize.detach() - x_crop
        commit_loss = torch.sum((diff ** 2) * loss_mask) / loss_mask.sum()
        loss = self._commitment_cost * commit_loss

        # Straight-through estimator
        quantize = x_crop + (quantize - x_crop).detach()

        # Expand patch-level indices/distances to per-frame
        # Indices: (B*P,) -> (B,P) -> (B,T_pad) -> crop
        patch_indices = torch.zeros(B * P, dtype=encoding_indices_valid.dtype, device=encoding_indices_valid.device)
        patch_indices[valid_patch_mask] = encoding_indices_valid
        encoding_indices = patch_indices.reshape(B, P).repeat_interleave(W, dim=1)[:, :end]  # (B, T)

        # Distances: (B*P,K) -> (B,P,K) -> (B,T_pad,K) -> crop
        patch_distances = torch.zeros(B * P, K, device=distances_valid.device)
        patch_distances[valid_patch_mask] = distances_valid
        distances = patch_distances.reshape(B, P, K).repeat_interleave(W, dim=1)[:, :end, :]  # (B, T, K)

        return quantize.contiguous(), encoding_indices, loss, distances
