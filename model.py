from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from src.model.smq import SMQModel
from src.model.utils import distance_joints
from src.model.eval_utils import evaluate_local_hungarian, evaluate_global_hungarian

class Trainer:
    
    """Trains SMQ and evaluates with MoF, Edit and F1 scores."""
    
    def __init__(self, in_channels, filters, num_layers, latent_dim, num_actions, 
                 num_joints, num_person, patch_size, kmeans, kmeans_metric, 
                 sampling_quantile, replacement_strategy, decay):
        """Builds the model and loss.

        Args:
            in_channels: Input feature channels per joint (C).
            filters: Base temporal conv width.
            num_layers: Number of dilated residual layers per stage.
            latent_dim: Latent channels per joint (Z).
            num_actions: Codebook size (K).
            num_joints: Number of joints (V).
            patch_size: Temporal window length for VQ (W).
            kmeans: Whether to initialize codebook with KMeans.
            kmeans_metric: Metric for KMeans init ('euclidean' or 'dtw').
            decay: EMA decay for codebook updates.
        """
        
        # Init model and loss
        self.model = SMQModel(in_channels = in_channels, filters = filters, 
                           num_layers = num_layers, latent_dim = latent_dim, 
                           num_actions = num_actions, num_joints = num_joints, 
                           num_person = num_person, patch_size = patch_size,
                           kmeans = kmeans, kmeans_metric = kmeans_metric, 
                           sampling_quantile = sampling_quantile, 
                           replacement_strategy = replacement_strategy, 
                           decay=decay)
        
        self.mse = nn.MSELoss(reduction='none')

    def train(self, save_dir, batch_gen, num_epochs, batch_size, 
              learning_rate, commit_weight, mse_loss_weight, device, 
              joint_distance_recons=True):
        
        # Train mode
        self.model.train()
        self.model.to(device)

        num_batches = batch_gen.num_batches(batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
        for epoch in range(num_epochs):

            pbar = tqdm(
            total=num_batches,
            desc=f"Training [Epoch {epoch+1}]",
            unit="batch",
            leave=False)

            epoch_rec_loss = 0.0
            epoch_commit = 0.0

            while batch_gen.has_next():
                batch_input, mask = batch_gen.next_batch(batch_size)
                batch_input, mask = batch_input.to(device), mask.to(device)

                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch_input,mask)

                # Reconstruction in joint-distance space
                if joint_distance_recons:
                    x, x_hat = distance_joints(batch_input), distance_joints(reconstructed)

                # Vanilla Reconstruction
                else :
                    x, x_hat = batch_input, reconstructed
                
                # Calculate loss
                rec_loss = mse_loss_weight * torch.mean(self.mse(x, x_hat))
                
                commit_loss = commit_weight * self.model.commit_loss
                loss = rec_loss + commit_loss

                # Backprop and update weights
                loss.backward()
                optimizer.step()

                epoch_rec_loss += rec_loss.item()
                epoch_commit += commit_loss.item()

                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            
            # Save Every 5 Epochs
            if (epoch + 1) % 5 == 0 :
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_dir / f"epoch-{epoch+1}.model")
                torch.save(optimizer.state_dict(), save_dir / f"epoch-{epoch+1}.opt")
            
            print("[epoch %d]: Reconstruction Loss = %f -- Commit Loss = %f" % 
                  (epoch + 1, epoch_rec_loss / num_batches, 
                   epoch_commit / num_batches))

    def eval(self, model_path, features_path, gt_path, mapping_file,
                epoch, vis , plot_dir, device) :
    
        # Eval mode
        self.model.eval()
        
        with torch.no_grad():
            # Load model
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))

            # --- Sequence Level Evaluation ---
            local_mof, local_edit, local_f1_vec, gt_all, prediction_all = evaluate_local_hungarian(
                model=self.model,
                features_path=features_path,
                gt_path=gt_path,
                mapping_file=mapping_file,
                epoch=epoch,
                device=device,
                verbose=True,
            )

            # --- Dataset Level Evaluation ---
            mof, edit, f1_vec, pr2gt = evaluate_global_hungarian(
                model=self.model,
                features_path=features_path,
                gt_path=gt_path,
                device=device,          
                mapping_file=mapping_file,
                epoch = epoch,
                vis=vis,                      
                plot_dir=plot_dir,
                gt_all=gt_all,
                prediction_all=prediction_all,
                verbose=True,
            )