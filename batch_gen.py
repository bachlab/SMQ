# =============================================================================
# batch_gen.py — Batch loading utilities for SMQ
# Adapted from MS-TCN: https://github.com/yabufarha/ms-tcn
# =============================================================================

import torch
import numpy as np
import random
import os
import math

random.seed(42)

class BatchGenerator(object):
    
    """Loads skeleton feature files and produces padded batches with masks.

    Expects .npy files shaped (C, T, V, M). Applies temporal subsampling,
    pads sequences to max length in batch, and returns (batch, mask).
    """

    def __init__(self, features_path, sample_rate, num_features, 
                 num_joints, num_person):
        
        self.list_of_examples = []
        self.index = 0
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.num_features = num_features
        self.num_joints = num_joints
        self.num_person = num_person

    def reset(self):
        """Resets index and reshuffles examples."""
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        """Returns whether more batches are available.

        Returns:
            bool: True if more data remains for this epoch.
        """
        return self.index < len(self.list_of_examples)

    def read_data(self):
        """Reads and shuffles file names.
        """
        self.list_of_examples = os.listdir(self.features_path)
        random.shuffle(self.list_of_examples)

    def num_batches(self, batch_size):
        """Returns the number of batches per epoch."""
        return math.ceil(len(self.list_of_examples) / batch_size)

    def next_batch(self, batch_size):
        """Loads the next batch and pads variable-length sequences.

        Args:
            batch_size (int): Number of samples to load.

        Returns:
            torch.Tensor: Batch tensor (N, C, T_max, V, M).
            torch.Tensor: Mask tensor, same shape, 1=valid, 0=padded.
        """
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        for vid in batch:
            try:
                features = np.load(os.path.join(self.features_path, vid))
                batch_input.append(features[:, ::self.sample_rate, :, :])
            except IOError:
                print(f'Error loading {vid}')

        length_of_sequences = list(map(lambda tensor: tensor.shape[1], batch_input))
        batch_input_tensor = torch.zeros(len(batch_input), self.num_features, max(length_of_sequences), self.num_joints, self.num_person, dtype=torch.float)
        mask = torch.zeros(len(batch_input), self.num_features, max(length_of_sequences), self.num_joints, self.num_person, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1], :, :] = torch.from_numpy(batch_input[i])
            mask[i, :, :np.shape(batch_input[i])[1], :, :] = torch.ones(np.shape(batch_input[i]))

        return batch_input_tensor, mask
