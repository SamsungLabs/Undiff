__all__ = ["FFCUnet"]


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from ffc_se_models.ffc_modules import FFCResNetBlock
from ffc_se_models.ffc_se_model import DecoupledSTFTPrediction, SigmaBlock
from ffc_se_models.nn_utils import AddSkipConn, ConcatSkipConn
from improved_diffusion.diffwave import DiffusionEmbedding
from improved_diffusion.unet import TimestepEmbedSequential, TimestepBlock


def build_conv_block(depth: int, conv, inplace: bool = True, **conv_kwargs):
    block = TimestepEmbedSequential(
        *[
            AddSkipConn(
                TimestepEmbedSequential(
                    nn.LeakyReLU(inplace=inplace), weight_norm(conv(**conv_kwargs))
                )
            )
            for _ in range(depth)
        ]
    )
    return block


def pad_to_divisible(x, modulo: int = 1, *args, **kwargs):
    """
    Pad STFT to be divisible by 2 ** scale_factor
    Padding is done by last dimension
    """

    if kwargs["mode"] == "zeros":
        kwargs["mode"] = "constant"

    dim_1 = (modulo - x.size(-1) % modulo) % modulo
    dim_2 = (modulo - x.size(-2) % modulo) % modulo
    return F.pad(x, (0, dim_1, 0, dim_2), *args, **kwargs)


class UNetBaseBlock(TimestepBlock):
    def __init__(
        self,
        start_block=nn.Identity(),
        downsample_block=nn.Identity(),
        net=nn.Identity(),
        upsample_block=nn.Identity(),
        end_block=nn.Identity(),
        use_connection: str = None,
    ):
        super().__init__()

        module_dict = TimestepEmbedSequential(
            OrderedDict(
                [
                    ("start_block", start_block),
                    ("downsample", downsample_block),
                    ("internal_block", net),
                    ("upsample", upsample_block),
                    ("end_block", end_block),
                ]
            )
        )

        self.model = self._wrap_connection(
            connection_type=use_connection, module=module_dict
        )

    @staticmethod
    def _wrap_connection(connection_type: str, module):
        if connection_type == "concat":
            module = ConcatSkipConn(module)
        elif connection_type == "add":
            module = AddSkipConn(module)

        return module

    def forward(self, x, t):
        return self.model(x, t)


class FFCUnet(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        scale_factor: int = 3,
        block_depth: tuple = (1, 1, 1),
        use_connection: str = None,
        use_full_ffc: bool = False,
        n_ffc: int = 9,
        fu_kernel: int = 1,
        ffc_conv_kernel: int = 3,
        ffc_global_ratio_in: tuple = (0.75, 0.75, 0.75, 0.75),
        ffc_global_ratio_out: tuple = (0.75, 0.75, 0.75, 0.75),
        padding_type: str = "reflect",
        stride: tuple = (2,),
        bias: bool = False,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        center: bool = True,
        fft_norm: str = "ortho",
        use_only_freq: bool = True,
        window: str = "hann_window",
        use_decoupled: bool = False,
        embedding_mode: str = "sigma",
    ):
        super().__init__()

        if embedding_mode == "time":
            self.t_embedding = DiffusionEmbedding(max_steps=200)
        elif embedding_mode == "sigma":
            self.sigma_embedding = SigmaBlock()

        self.scale_factor = scale_factor
        self.padding_type = padding_type
        assert len(block_depth) == self.scale_factor
        assert len(ffc_global_ratio_in) == self.scale_factor + 1
        assert len(ffc_global_ratio_out) == self.scale_factor + 1

        extended_channels = out_channels * (2**scale_factor)
        self.stft_params = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "center": center,
        }
        self.window = self._get_window(window)

        model = TimestepEmbedSequential(
            *[
                FFCResNetBlock(
                    extended_channels,
                    extended_channels,
                    alpha_in=ffc_global_ratio_in[-1],
                    alpha_out=ffc_global_ratio_out[-1],
                    kernel_size=ffc_conv_kernel,
                    padding_type=padding_type,
                    bias=bias,
                    fu_kernel=fu_kernel,
                    fft_norm=fft_norm,
                    use_only_freq=use_only_freq,
                    use_film=False,
                )
                for _ in range(n_ffc)
            ]
        )

        for i in range(self.scale_factor, 0, -1):
            cur_ch = out_channels * (2**i)
            start_block = (
                TimestepEmbedSequential(
                    *[
                        FFCResNetBlock(
                            cur_ch // 2,
                            cur_ch // 2,
                            alpha_in=ffc_global_ratio_in[i - 1],
                            alpha_out=ffc_global_ratio_out[i - 1],
                            kernel_size=ffc_conv_kernel,
                            padding_type=padding_type,
                            bias=bias,
                            fu_kernel=fu_kernel,
                            fft_norm=fft_norm,
                            use_only_freq=use_only_freq,
                            use_film=False,
                        )
                        for _ in range(block_depth[i - 1])
                    ]
                )
                if use_full_ffc
                else build_conv_block(
                    depth=block_depth[i - 1],
                    conv=nn.Conv2d,
                    inplace=False,
                    in_channels=cur_ch // 2,
                    out_channels=cur_ch // 2,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    padding_mode=padding_type,
                    bias=bias,
                )
            )
            end_block = (
                TimestepEmbedSequential(
                    *[
                        FFCResNetBlock(
                            cur_ch // 2,
                            cur_ch // 2,
                            alpha_in=ffc_global_ratio_in[i - 1],
                            alpha_out=ffc_global_ratio_out[i - 1],
                            kernel_size=ffc_conv_kernel,
                            padding_type=padding_type,
                            bias=bias,
                            fu_kernel=fu_kernel,
                            fft_norm=fft_norm,
                            use_only_freq=use_only_freq,
                            use_film=False,
                        )
                        for _ in range(block_depth[i - 1])
                    ]
                )
                if use_full_ffc
                else build_conv_block(
                    depth=block_depth[i - 1],
                    conv=nn.Conv2d,
                    inplace=False,
                    in_channels=cur_ch // 2,
                    out_channels=cur_ch // 2,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    padding_mode=padding_type,
                    bias=bias,
                )
            )
            model = TimestepEmbedSequential(
                OrderedDict(
                    [
                        (
                            "UNetBaseBlock",
                            UNetBaseBlock(
                                start_block=start_block,
                                downsample_block=nn.Conv2d(
                                    cur_ch // 2,
                                    cur_ch,
                                    kernel_size=1,
                                    stride=tuple(stride),
                                    padding_mode=padding_type,
                                    bias=bias,
                                ),
                                net=model,
                                upsample_block=TimestepEmbedSequential(
                                    nn.Upsample(scale_factor=tuple(stride)),
                                    nn.Conv2d(
                                        cur_ch, cur_ch // 2, kernel_size=1, bias=bias
                                    ),
                                ),
                                end_block=end_block,
                                use_connection=use_connection,
                            ),
                        ),
                        (
                            "ConcatConv",
                            nn.Conv2d(cur_ch, cur_ch // 2, kernel_size=1, bias=bias)
                            if use_connection == "concat"
                            else nn.Identity(),
                        ),
                    ]
                )
            )

        class WrappedModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.net = model

            def forward(self, x, t):
                x = x.permute(0, 3, 1, 2)

                dim_2, dim_1 = x.size()[-2:]
                x = pad_to_divisible(x, modulo=2**scale_factor, mode=padding_type)

                out = self.net(x, t)[..., :dim_2, :dim_1].permute(0, 2, 3, 1)

                return out

        start_channels = 1 if use_decoupled else 2
        final_channels = 4 if use_decoupled else 2

        model = TimestepEmbedSequential(
            nn.Conv2d(
                start_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                padding_mode=padding_type,
                bias=bias,
            ),
            model,
            nn.Conv2d(
                out_channels,
                final_channels,
                kernel_size=7,
                padding=3,
                padding_mode=padding_type,
                bias=bias,
            ),
        )

        model = WrappedModel(model)

        self.model = (
            DecoupledSTFTPrediction(model=model, bounding_func_name="sigmoid")
            if use_decoupled
            else model
        )

    def _get_window(self, window_name: str):
        method_name = window_name
        window = None

        if method_name == "hann_window":
            window = torch.hann_window(self.stft_params["win_length"], periodic=False)
        elif method_name == "hamming_window":
            window = torch.hamming_window(
                self.stft_params["win_length"], periodic=False
            )

        return window

    def forward(self, x, t):
        x = x.squeeze(1)
        original_dim = x.size(-1)

        if self.window is not None:
            self.window = self.window.to(x.device)

        stft_representation = torch.view_as_real(
            torch.stft(x, window=self.window, return_complex=True, **self.stft_params)
        )

        if hasattr(self, "t_embedding"):
            emb = self.t_embedding(t)
        elif hasattr(self, "sigma_embedding"):
            emb = self.sigma_embedding(t)

        reconstructed_stft = self.model(stft_representation, emb)

        reconstructed_batch = torch.istft(
            torch.view_as_complex(reconstructed_stft.contiguous()),
            length=original_dim,
            window=self.window,
            return_complex=False,
            **self.stft_params,
        )

        return reconstructed_batch.unsqueeze(1)
