import torch.nn as nn
import torch
import math
from typing import Sequence, Tuple, Optional


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvTransposeBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, output_padding=0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                output_padding=output_padding,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_length: int,
        latent_dim: int,
        filters: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        pool_sizes: Sequence[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = [1] + list(filters)
        layers = []
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.filters = tuple(filters)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)
        self.pool_sizes = tuple(strides)
        self.dropout = dropout

        current_length = input_length
        self.track_len = [current_length]
        self.pre_pool_len = []

        for in_channels, out_channels, kernel_size, stride, pool_size in zip(
            channels[:-1], channels[1:], kernel_sizes, strides, pool_sizes
        ):
            layers.append(ConvBNRelu(in_channels, out_channels, kernel_size, stride))
            current_length = math.floor(
                (current_length + 2 * (kernel_size // 2) - kernel_size) / stride + 1
            )
            self.track_len.append(current_length)
            layers.append(ConvBNRelu(out_channels, out_channels, kernel_size, stride))
            current_length = math.floor(
                (current_length + 2 * (kernel_size // 2) - kernel_size) / stride + 1
            )
            self.track_len.append(current_length)
            self.pre_pool_len.append(current_length)
            layers.append(nn.MaxPool1d(pool_size))
            current_length = math.floor((current_length - pool_size) / pool_size + 1)
            self.track_len.append(current_length)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(current_length * channels[-1], self.latent_dim * 2))
        layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        self._last_convolutional_shape = (channels[-1], current_length)

    def forward(self, x):
        """Remove flattening of single dim"""
        x = self.net(x).view(-1, self.latent_dim, 2)
        mu = x[..., 0]
        log_var = x[..., 1]
        return mu, log_var

    @property
    def last_convolutional_shape(self):
        return self._last_convolutional_shape


class CNNDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        expansive_shape: Tuple[int, int],
        filters: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        upsample_sizes: Sequence[int],
        output_paddings: Optional[Sequence[Tuple[int, int]]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = list(filters) + [1]
        current_length = expansive_shape[0] * expansive_shape[1]
        self.track_len = []
        self.expansive_shape = expansive_shape
        self.expansive_layer = nn.Sequential(
            nn.Linear(latent_dim, current_length),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        layers = []
        self.track_len.append(expansive_shape[-1])

        for (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            upsample_size,
            output_pad,
        ) in zip(
            channels[:-1],
            channels[1:],
            kernel_sizes,
            strides,
            upsample_sizes,
            output_paddings,
        ):
            layers.append(nn.Upsample(upsample_size))
            self.track_len.append(upsample_size)
            layers.append(
                ConvTransposeBlock(
                    in_channels, out_channels, kernel_size, stride, output_pad[0]
                )
            )
            self.track_len.append(
                self.devconv_output_len(
                    self.track_len[-1],
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    output_pad[0],
                )
            )

            layers.append(
                ConvTransposeBlock(
                    out_channels, out_channels, kernel_size, stride, output_pad[1]
                )
            )
            self.track_len.append(
                self.devconv_output_len(
                    self.track_len[-1],
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    output_pad[1],
                )
            )
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def devconv_output_len(l_in, kernel_size, stride, padding, output_padding=0):
        """
        Output length from a single deconvolutional layer. No dialation assumed
        Lout=(Lin−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
        """
        return (
            (l_in - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
        )

    @classmethod
    def from_encoder(cls, encoder: CNNEncoder):
        latent_dim = encoder.latent_dim
        expansive_shape = encoder.last_convolutional_shape

        idx = -1
        output_paddings = []
        for kernel_size, stride, upsample_size in zip(
            reversed(encoder.kernel_sizes),
            reversed(encoder.strides),
            reversed(encoder.pre_pool_len),
        ):
            length = upsample_size
            idx -= 1
            length = cls.devconv_output_len(
                length, kernel_size, stride, kernel_size // 2
            )
            idx -= 1
            if length != encoder.track_len[idx]:
                pad1 = encoder.track_len[idx] - length
                length += pad1
            else:
                pad1 = 0
            idx -= 1
            length = cls.devconv_output_len(
                length, kernel_size, stride, kernel_size // 2
            )
            if length != encoder.track_len[idx]:
                pad2 = encoder.track_len[idx] - length
                length += pad2
            else:
                pad2 = 0
            output_paddings.append((pad1, pad2))

        return cls(
            latent_dim=latent_dim,
            expansive_shape=expansive_shape,
            filters=tuple(reversed(encoder.filters)),
            kernel_sizes=tuple(reversed(encoder.kernel_sizes)),
            strides=tuple(reversed(encoder.strides)),
            upsample_sizes=tuple(reversed(encoder.pre_pool_len)),
            output_paddings=output_paddings,
        )

    def forward(self, x):
        x = self.expansive_layer(x)
        x = x.view(-1, self.expansive_shape[0], self.expansive_shape[1])
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.latent_dim

    @staticmethod
    def sample_posterior(mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample_posterior(mu, log_var)
        return self.decoder(z), mu, log_var


if __name__ == "__main__":
    pass
