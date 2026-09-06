"""
Multilayer Perceptron (MLP) Baseline Model adhering to Part 2 of the rubric.
Flattens input images into 1D vectors and processes through fully connected layers.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Baseline Multilayer Perceptron.
    Demonstrates the limitations of fully connected architectures for image tasks:
    - Loss of 2D spatial correlations and translation invariance
    - Parameter explosion as image resolution scales
    """

    def __init__(
        self,
        input_shape: tuple = (3, 32, 32),
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 50,
        activation: str = "relu",
        dropout: float = 0.0,
        init_type: str = "kaiming"
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_shape = input_shape
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.num_classes = num_classes

        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(0.1)
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final classification head
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self._init_weights(init_type)

    def _init_weights(self, init_type: str) -> None:
        """Applies weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize dynamically if needed to target shape, or flatten
        if x.shape[1:] != self.input_shape:
            x = nn.functional.interpolate(
                x,
                size=(self.input_shape[1], self.input_shape[2]),
                mode="bilinear",
                align_corners=False
            )
        x = torch.flatten(x, start_dim=1)
        return self.network(x)
