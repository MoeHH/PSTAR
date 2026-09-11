"""Fixed 2D sinusoidal positional encoding over grid (x, y) coordinates for PSTAR."""
import torch
from torch import nn
import numpy as np
from config import config


class Sinusoidal2DPE(nn.Module):
    """
    Fixed (non-learnable) 2D positional encoding for grid coordinates (x, y).
    """
    def __init__(self, dim, grid_size=None):
        super().__init__()
        self.dim = int(dim)

        if grid_size is None:
            grid_size = config["grid_size"]
        self.grid_size = grid_size

        if self.dim % 4 != 0:
            raise ValueError(f"Sinusoidal2DPE requires dim % 4 == 0, got dim={self.dim}.")

        height = int(self.grid_size[0])
        width = int(self.grid_size[1])

        quarter = self.dim // 4

        pe = torch.zeros(height, width, self.dim)

        y = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        x = torch.arange(width, dtype=torch.float32).unsqueeze(1)   # [W, 1]

        div_term = torch.exp(
            torch.arange(quarter, dtype=torch.float32) * (-np.log(10000.0) / quarter)
        )  # [quarter]

        y_arg = y * div_term  # [H, quarter]
        x_arg = x * div_term  # [W, quarter]

        pe[:, :, 0:quarter] = torch.sin(y_arg).unsqueeze(1).expand(-1, width, -1)
        pe[:, :, quarter:2 * quarter] = torch.cos(y_arg).unsqueeze(1).expand(-1, width, -1)

        pe[:, :, 2 * quarter:3 * quarter] = torch.sin(x_arg).unsqueeze(0).expand(height, -1, -1)
        pe[:, :, 3 * quarter:4 * quarter] = torch.cos(x_arg).unsqueeze(0).expand(height, -1, -1)

        self.register_buffer("pe", pe)

    def forward(self, input_tensor):
        padding_value = config["padding_value"]
        one_indexed = bool(config.get("one_indexed", True))

        padded_rows = input_tensor[:, :, 0].eq(padding_value)  # [B, seq_len]

        x_coords = input_tensor[:, :, 1].long()
        y_coords = input_tensor[:, :, 2].long()

        if one_indexed:
            x_coords = x_coords - 1
            y_coords = y_coords - 1

        x_coords = torch.clamp(x_coords, 0, self.grid_size[1] - 1)
        y_coords = torch.clamp(y_coords, 0, self.grid_size[0] - 1)

        pos_encoding = self.pe[y_coords, x_coords]  # [B, seq_len, dim]

        if padded_rows.any():
            pos_encoding = pos_encoding.masked_fill(padded_rows.unsqueeze(-1), 0.0)

        return pos_encoding
