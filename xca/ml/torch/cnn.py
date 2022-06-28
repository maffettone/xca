import torch
import torch.nn as nn
from typing import Sequence
from xca.ml.torch import Expression


class ConvSubUnit(nn.Module):
    """Convolutional subunit contianing Conv, Leaky Relu, Average pool, and batch norm"""

    def __init__(
        self, *, in_channels, out_channels, kernel_size, stride, pool_size, alpha
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.LeakyReLU(negative_slope=alpha),
            nn.AvgPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class GenericCNNModel(nn.Module):
    def __init__(
        self,
        *,
        filters: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        pool_sizes: Sequence[int],
        n_classes: int,
        ReLU_alpha: float = 0.0,
        dense_dims: Sequence[int] = (),
        dense_dropout: float = 0.0,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pool_sizes = pool_sizes
        self.n_classes = n_classes
        self.ReLU_alpha = ReLU_alpha
        self.dense_dims = dense_dims
        self.dense_dropout = dense_dropout

        channels = [1] + list(filters)
        layers = [
            ConvSubUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pool_size=pool_size,
                alpha=ReLU_alpha,
            )
            for in_channels, out_channels, kernel_size, stride, pool_size in zip(
                channels[:-1], channels[1:], kernel_sizes, strides, pool_sizes
            )
        ]
        # Linearize by mean and add fully connected layers
        layers += [
            Expression(lambda x: x.mean(-1)),  # Mean over spectral dimension
            nn.Dropout(dense_dropout),
        ]
        dense_dims = [channels[-1]] + list(dense_dims)
        for in_dim, out_dim in zip(dense_dims[:-1], dense_dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dense_dropout)]
        layers += [nn.Linear(dense_dims[-1], n_classes)]

        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    @property
    def hparams(self) -> dict:
        return {
            key: getattr(self, key)
            for key in (
                "filters",
                "kernel_sizes",
                "strides",
                "pool_sizes",
                "n_classes",
                "ReLU_alpha",
                "dense_dims",
                "dense_dropout",
            )
        }

    def forward(self, x):
        if self.n_classes == 1:
            return self.net(x)[..., 0]
        return self.net(x)


class EnsembleCNN(nn.Module):
    def __init__(self, ensemble_size: int, *args, **kwargs):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.sub_nets = nn.ModuleList(
            [GenericCNNModel(*args, **kwargs) for _ in range(ensemble_size)]
        )

    @property
    def hparams(self):
        d = {"ensemble_size": self.ensemble_size}
        d.update(self.sub_nets[0].hparams)
        return d

    def forward(self, x):
        ys = [net(x) for net in self.sub_nets]
        return torch.stack(ys).mean(dim=0)
