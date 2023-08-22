__all__ = [
    "MOSNet",
    "LSD",
    "SiSNR",
]

import os
from abc import ABC, abstractmethod

import hydra.utils
import numpy as np
import torch
import torchaudio
from hydra.core.hydra_config import HydraConfig

from improved_diffusion.metric_nets import Wav2Vec2MOS


class Metric(ABC):
    name = "Abstract Metric"

    def __init__(self, num_splits=5, device="cuda", big_val_size=500):
        self.num_splits = num_splits
        self.device = device
        self.val_size = None
        self.result = dict()
        self.big_val_size = big_val_size

    @abstractmethod
    def better(self, first, second):
        pass

    @abstractmethod
    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        pass

    def compute(self, samples, real_samples, epoch_num, epoch_info):
        self._compute(samples, real_samples, epoch_num, epoch_info)
        self.result["val_size"] = samples.shape[0]

        if "best_mean" not in self.result or self.better(
            self.result["mean"], self.result["best_mean"]
        ):
            self.result["best_mean"] = self.result["mean"]
            self.result["best_std"] = self.result["std"]
            self.result["best_epoch"] = epoch_num

    def save_result(self, epoch_info):
        metric_name = self.name
        for key, value in self.result.items():
            epoch_info[f"metrics_{key}/{metric_name}"] = value


def get_power_spectrum(wav, n_fft: int, hop_length: int):
    spectrum = torch.stft(
        wav.view(wav.size(0), -1),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    ).abs()
    return spectrum.transpose(-1, -2)  # (B, T, F)


class LSD(Metric):
    name = "LSD"

    def __init__(self, sampling_rate: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sampling_rate
        self.hop_length = int(self.sr / 100)
        self.n_fft = int(2048 / (44100 / self.sr))

    def better(self, first, second):
        return first < second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        spectrum_hat = get_power_spectrum(
            samples, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectrum = get_power_spectrum(
            real_samples, n_fft=self.n_fft, hop_length=self.hop_length
        )

        lsd = torch.log10(spectrum**2 / (spectrum_hat**2 + 1e-9) + 1e-9) ** 2
        lsd = torch.sqrt(lsd.mean(-1)).mean(-1)

        lsd = lsd.cpu().numpy()
        self.result["mean"] = np.mean(lsd)
        self.result["std"] = np.std(lsd)


class SiSNR(Metric):
    name = "SiSNR"

    def better(self, first, second):
        return first > second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        alpha = (samples * real_samples).sum(-1, keepdims=True) / (
            real_samples.square().sum(-1, keepdims=True) + 1e-9
        )
        real_samples_scaled = alpha * real_samples
        e_target = real_samples_scaled.square().sum(-1)
        e_res = (real_samples_scaled - samples).square().sum(-1)
        sisnr = 10 * torch.log10(e_target / (e_res + 1e-9)).cpu().numpy()

        self.result["mean"] = np.mean(sisnr)
        self.result["std"] = np.std(sisnr)


class MOSNet(Metric):
    name = "MOSNet"

    def __init__(
        self,
        weights_path: str,
        pretrained_path: str,
        sr=22050,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if HydraConfig.initialized():
            orig_cwd = hydra.utils.get_original_cwd()
            weights_path = os.path.join(orig_cwd, weights_path)
            pretrained_path = os.path.join(orig_cwd, pretrained_path)

        self.mos_net = Wav2Vec2MOS(weights_path, pretrained_path, device=device)
        self.sr = sr
        self.device = device

    def better(self, first, second):
        return first > second

    def _compute_per_split(self, split):
        return self.mos_net.calculate(split)

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=required_sr
        ).to(samples.device)

        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]

        splits = [
            samples[i : i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]
        self.result["mean"] = np.mean(fid_per_splits)
        self.result["std"] = np.std(fid_per_splits)
