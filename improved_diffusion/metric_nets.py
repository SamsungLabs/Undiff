from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix) :]] = weights[key]
    return result


class Wav2Vec2MOS(torch.nn.Module):
    sample_rate = 16_000

    def __init__(self, path, pretrained_path: str, freeze=True, device: str = "cuda"):
        super().__init__()
        rel_pretrained_path = pretrained_path
        self.encoder = Wav2Vec2Model.from_pretrained(rel_pretrained_path)
        self.freeze = freeze

        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.load_state_dict(
            extract_prefix(
                "model.", torch.load(path, map_location=device)["state_dict"]
            )
        )
        self.eval()
        self.to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(rel_pretrained_path)

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]  # [Batch, time, feats]
        x = self.dense(x)  # [batch, time, 1]
        x = x.mean(dim=[1, 2], keepdims=True)  # [batch, 1, 1]
        return x

    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()

    def to_device(self, device):
        self.encoder.to(device)
        self.dense.to(device)

    def calculate(self, samples):
        pred_mos = []
        self.to_device(samples[0].device)
        for s in samples:
            x = self.processor(
                s.cpu(),
                return_tensors="pt",
                padding=True,
                sampling_rate=self.sample_rate,
            ).input_values
            with torch.no_grad():
                res = self.forward(x.to(s.device)).mean()
            pred_mos.append(res.item())
        return np.mean(pred_mos)
