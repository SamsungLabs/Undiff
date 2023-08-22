__all__ = ["FFCMelModel"]

import torch
import torch.nn as nn

from ffc_se_models.ffc_modules import FFCResNetBlock
from improved_diffusion.diffwave import DiffusionEmbedding
from improved_diffusion.unet import TimestepEmbedSequential


class FFCMelModel(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        ch: int = 32,
        depth: int = 4,
        n_ffc: int = 9,
        noise_schedule_len: int = 200,
        ffc_kwargs=None,
    ):
        super().__init__()

        self.t_embedding = DiffusionEmbedding(noise_schedule_len)

        self.start_conv = nn.Conv2d(in_dim, ch, kernel_size=3, padding=1)
        encoder = []

        # downsampling encoder blocks
        for _ in range(depth):
            ffc_block = [
                FFCResNetBlock(in_channels=ch, out_channels=ch, **ffc_kwargs)
                for _ in range(n_ffc)
            ]

            down_ffc_block = TimestepEmbedSequential(
                *ffc_block,
                nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU()
            )

            encoder.append(down_ffc_block)

            ch *= 2

        self.encoder = TimestepEmbedSequential(*encoder)

        decoder = []

        for i in range(depth):
            ffc_block = [
                FFCResNetBlock(
                    in_channels=ch, out_channels=ch, alpha_in=0.75, alpha_out=0.75
                )
                for _ in range(n_ffc)
            ]

            up_ffc_block = TimestepEmbedSequential(
                *ffc_block,
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )

            decoder.append(up_ffc_block)

            ch //= 2

        self.decoder = TimestepEmbedSequential(*decoder)

        self.final_conv = nn.Conv2d(ch, in_dim, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = self.start_conv(x)

        t_emb = self.t_embedding(t)

        x = self.encoder(x, t_emb)

        x = self.decoder(x, t_emb)

        x = self.final_conv(x)

        return x
