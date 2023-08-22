import random

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    use_full_spec=False,
    return_mel_and_spec=False,
    use_log_normalize=True,
):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )

    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    result = spectral_normalize_torch(spec)

    if not use_full_spec:
        mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
        if use_log_normalize:
            mel = spectral_normalize_torch(mel)
        result = mel.squeeze()

        if return_mel_and_spec:
            if use_log_normalize:
                spec = spectral_normalize_torch(spec)
            return result, spec
    return result


def cut_audio_segment(audio: torch.Tensor, segment_size: int):
    if audio.size(1) >= segment_size:
        max_audio_start = audio.size(1) - segment_size
        audio_start = random.randint(0, max_audio_start)
        audio = audio[:, audio_start : audio_start + segment_size]
    else:
        audio = torch.nn.functional.pad(
            audio, (0, segment_size - audio.size(1)), "constant"
        )

    return audio
