import torch.nn as nn

from src.model.motion_quantizer import SkeletonMotionQuantizer
from src.model.utils import process_mask
from src.model.ms_tcn import MultiStageModel

class SMQModel(nn.Module):
    """
    TCN Autoencoder with SMQ (Patch-based VQ).

    Encodes each disentangeled joint with TCN, concatenates all joint latents
    per frame into a single token, applies temporal vector quantization over fixed windows,
    then decodes back to skeleton space.
    """
    def __init__(self, in_channels=6, filters=64, num_layers = 3, latent_dim=16, 
                 num_actions=8, num_joints=22, num_person=1, patch_size=50, kmeans=False, 
                 kmeans_metric='euclidean', sampling_quantile=0.5, replacement_strategy="representative", 
                 decay=0.5):
        
        super(SMQModel, self).__init__()

        self.latent_dim = latent_dim
        self.num_joints = num_joints
        self.num_person = num_person

        self.commit_loss = None
        self.latent = None
        self.indices = None

        # Encoder
        self.encoder = MultiStageModel(num_layers = num_layers, num_f_maps = filters, 
                                       dim = in_channels, target_dim1 = int(self.latent_dim/2), 
                                       target_dim2 = self.latent_dim)
        
        # VQ
        self.vq = SkeletonMotionQuantizer(num_embeddings = num_actions, embedding_dim = latent_dim * num_joints * num_person, 
                      window = patch_size, commitment_cost = 1.0, decay=decay, eps=1e-5,
                      threshold_ema_dead_code=10, sampling_quantile=sampling_quantile, 
                      replacement_strategy = replacement_strategy,kmeans=kmeans, 
                      kmeans_metric =kmeans_metric)
        
        # Decoder
        self.decoder = MultiStageModel(num_layers = num_layers, num_f_maps = filters, 
                                       dim = self.latent_dim, target_dim1 = int(self.latent_dim/2), 
                                       target_dim2 = in_channels)

    def forward(self, x, mask):
        """
        Reconstruct skeleton sequences via joint-disentangled 
        autoencoding + SMQ (Patch-based VQ).
            
        Args:
            x:    (N, C, T, V, M)  skeleton features
            mask: (N, C, T, V, M)  1=valid, 0=padded
        Returns:
            out:  (N, C, T, V, M)  reconstructed skeleton features
        """
        
        N, C, T, V, M = x.size()

        # Pack person+joint dims : (N, C, T, V, M) -> (N*M*V, C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, C, T)
        mask = mask.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, C, T)

        # Encode joint-disentangled latents: (N*M*V, C, T) -> (N*M*V, latent_dim, T)
        self.latent = self.encoder(x, mask)

        # Repack into per-frame skeleton : (N*M*V, latent_dim, T) -> (N, T, V, M, latent_dim)
        latent = self.latent.view(N * M, V, self.latent_dim, T).contiguous()  # (N*M, V, latent_dim, T)
        latent = latent.permute(0, 3, 1, 2)                                   # (N*M, T, V, latent_dim)
        latent = latent.reshape(N, M, T, V, self.latent_dim)                  # (N, M, T, V, latent_dim) 
        latent = latent.permute(0, 2, 3, 1, 4)                                # (N, T, V, M, latent_dim)

        # Concat joints for quantization : (N, T, V, M, latent_dim) -> (N, T, V*M*latent_dim)
        latent = latent.reshape(N, T, -1)

        # Build mask matching VQ input: (N, T, V*M*latent_dim)
        vq_mask = process_mask(mask, batch_size=N, num_person=M, num_joints=self.num_joints, seq_length=T, num_features_per_joint=self.latent_dim)

        # Quantization - VQ expects token dim = V * M * latent_dim.
        quantized, self.indices, self.commit_loss, _ = self.vq(latent, vq_mask)

        # Prepare for decoder : (N, T, V*M*latent_dim) -> (N*M*V, latent_dim, T)
        quantized = quantized.reshape(N, T, V, M * self.latent_dim)                          # (N, T, V, M*latent_dim)
        quantized = quantized.reshape(N, T, V, M, self.latent_dim).permute(0, 3, 4, 1, 2)    # (N, M, latent_dim, T, V)
        quantized = quantized.reshape(N*M, self.latent_dim, T, V)                            # (N*M, latent_dim, T, V)
        quantized = quantized.permute(0, 3, 1, 2)                                            # (N*M, V, latent_dim, T)
        quantized = quantized.reshape(N*M*V, self.latent_dim, T)                             # (N*M*V, latent_dim, T)
        
        # Decode : (N*M*V, latent_dim, T) -> (N*M*V, C, T)
        decoded = self.decoder(quantized, mask)
        
        # Unpack back to original skeleton layout: (N, C, T, V, M)
        decoded = decoded.reshape(N * M, V, C, T)
        decoded = decoded.reshape(N, M, V, C, T)
        out = decoded.permute(0, 3, 4, 2, 1)

        return out