import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiffWave"]

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def exists(val):
    return val is not None


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx).view(-1, 1)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)


class RandomFourierFeatures(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.register_buffer(
            "gaussian", torch.randn((1, N), requires_grad=False), persistent=True
        )

    def forward(self, sigmas):
        resized_sigmas = sigmas[..., None].expand(-1, self.N)
        cosines = torch.cos(2.0 * math.pi * resized_sigmas * self.gaussian)
        sines = torch.sin(2.0 * math.pi * resized_sigmas * self.gaussian)
        return torch.cat([cosines, sines], dim=1)


class SigmaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.rff = RandomFourierFeatures(64)
        self.proj1 = nn.Linear(128, 512)
        self.proj2 = nn.Linear(512, 512)

    def forward(self, cond):
        rescaled = torch.log10(cond)
        emb = self.rff(rescaled)
        out = self.proj1(emb)
        out = silu(out)
        out = self.proj2(out)
        out = silu(out)
        return out


class FiLM(nn.Module):
    def forward(self, x, condition):
        gamma, beta = torch.chunk(condition, chunks=2, dim=1)
        gamma = gamma.expand_as(x)
        beta = beta.expand_as(x)
        return (gamma * x) + beta


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_mels,
        residual_channels,
        dilation,
        index,
        attention,
        use_film,
        kernel_size,
        se_num,
        uncond=True,
    ):
        """
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        """
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(
            512, residual_channels * 2 if use_film else residual_channels
        )
        if not uncond:
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else:
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

        assert attention in ["se_skip", "se_res", "se_x", None]
        if attention == "se_skip":
            self.se_skip = SE_Block(residual_channels, se_num)

        if attention == "se_res":
            self.se_res = SE_Block(residual_channels, se_num)

        if attention == "se_x":
            self.se_x = SE_Block(residual_channels, se_num)

        if use_film:
            self.film = FiLM()

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or (
            conditioner is not None and self.conditioner_projection is not None
        )
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)

        if hasattr(self, "film"):
            y = self.film(x, diffusion_step)

        else:
            y = x + diffusion_step

        if self.conditioner_projection is None:
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)

        if hasattr(self, "se_skip"):
            skip = self.se_skip(skip)

        if hasattr(self, "se_res"):
            residual = self.se_res(residual)

        if hasattr(self, "se_x"):
            x = self.se_x(x)

        x = (x + residual) / math.sqrt(2.0)

        return x, skip


class DiffWave(nn.Module):
    def __init__(
        self,
        residual_layers,
        residual_channels,
        noise_schedule_len,
        attention,
        embedding_type,
        kernel_size,
        use_film,
        se_num,
        unconditional=True,
        n_mels=80,
        dilation_cycle_length=12,
    ):
        super().__init__()
        self.input_projection = Conv1d(1, residual_channels, 1)

        if embedding_type == "time":
            self.diffusion_embedding = DiffusionEmbedding(noise_schedule_len)
        elif embedding_type == "sigma":
            self.diffusion_embedding = SigmaBlock()
        else:
            raise NotImplementedError

        if unconditional:
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(n_mels)

        self.residual_layers = nn.ModuleList()
        for i in range(residual_layers):
            self.residual_layers.append(
                ResidualBlock(
                    n_mels,
                    residual_channels,
                    2 ** (i % dilation_cycle_length),
                    uncond=unconditional,
                    index=i,
                    attention=attention,
                    use_film=use_film,
                    kernel_size=kernel_size,
                    se_num=se_num,
                )
            )

        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self, x, diffusion_step=torch.tensor(100, dtype=torch.int32), spectrogram=None
    ):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or (
            spectrogram is not None and self.spectrogram_upsampler is not None
        )
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
