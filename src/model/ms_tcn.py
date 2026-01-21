# =============================================================================
# ms_tcn.py — Two stage TCN model
# Adapted from MS-TCN: https://github.com/yabufarha/ms-tcn
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F
import copy

class MultiStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, target_dim1, target_dim2):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, target_dim1)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, target_dim1, target_dim2)

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        out = self.stage2(out, mask)
        return out


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, target_dim):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, target_dim, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]