import copy
import itertools
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm


def get_tensor_from_list(audio_list):
    return torch.cat(audio_list, dim=0)


def log_results(results_dir, res, name="metrics"):
    print(res)
    file_exp_res = os.path.join(results_dir, f"{name}.txt")
    with open(file_exp_res, "w+") as f:
        for k, v in res.items():
            print(f"{k}/mean: {v[0]:.3f}", file=f)
            print(f"{k}/std: {v[1]:.3f}", file=f)


def create_state_dict_from_ema(state_dict, model, ema_params):
    ema_state_dict = copy.deepcopy(state_dict)
    for i, (name, _) in enumerate(model.named_parameters()):
        ema_state_dict[name] = ema_params[i]

    return ema_state_dict


def remove_prefix_from_state_dict(state_dict, j: int = 1):
    new_state_dict = OrderedDict()
    for k, _ in state_dict.items():
        tokens = k.split(".")
        new_state_dict[".".join(tokens[j:])] = state_dict[k]

    return new_state_dict


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_metric_result(pred_tensor, metrics, real_tensor=None):
    res = {}
    for metric in metrics:
        metric.val_size = pred_tensor.size(0)
        metric.compute(pred_tensor, real_tensor, 0, res)
        metric.save_result(res)
    return res


def calculate_all_metrics(wavs, metrics, n_max_files=None, reference_wavs=None):
    scores = {metric.name: [] for metric in metrics}
    if reference_wavs is None:
        reference_wavs = wavs
    for x, y in tqdm(
        itertools.islice(zip(wavs, reference_wavs), n_max_files),
        total=n_max_files if n_max_files is not None else len(wavs),
        desc="Calculating metrics",
    ):
        try:
            x = x.view(1, 1, -1)
            y = y.view(1, 1, -1)
            for metric in metrics:
                metric._compute(x, y, None, None)
                scores[metric.name] += [metric.result["mean"]]
        except Exception:
            pass
    scores = {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
    return scores
