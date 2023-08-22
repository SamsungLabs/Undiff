# UnDiff: Unsupervised Voice Restoration with Unconditional Diffusion Model

This is an official implementation of paper [UnDiff: Unsupervised Voice Restoration with Unconditional Diffusion Model](https://arxiv.org/abs/2306.00721) (Interspeech 2023)

## Installation

Clone the repo and install requirements:

```
mamba env create -f env.yaml
```

We use mamba-forge, but it also should work with conda as well.
Note that this environment uses Pytorch 2.0 with Cuda 11.8 latest stable release.

## Checkpoints

You can find checkpoints for Diffwave and FFC-AE models under Releases tab (`diffwave.ckpt` and `ffc_ae.ckpt`).
Download them to `weights` folder.

The contents of `weights/fb_w2v2_pretrained` are configs and checkpoints required for
[`transformers.Wav2Vec2Processor`](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor)
and [`transformers.Wav2Vec2Model`](https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Model)

The checkpoint for `transformers.Wav2Vec2Model` presented as 
`pytorch_model.bin` in Checkpoints tab should be downloaded and placed in `weights/fb_w2v2_pretrained`.

You would need Wav2Vec2Mos checkpoint to compute MOS metric. Download it by running ```download_extract_weights.sh```:
```
chmod +x download_extract_weights.sh
./download_extract_weights.sh
```

```
weights/
    ├── diffwave.ckpt
    ├── ffc_ae.ckpt
    ├── wave2vec2mos.pth
    └── fb_w2v2_pretrained
```

## Configs
Configs for inference can be found in ```configs``` directory:

```
configs/
├── diffusion
│   └── gaussian_diffusion.yaml
├── inference_cfg.yaml
├── model
│   ├── diffwave.yaml
│   └── ffc_ae.yaml
└── task
    ├── bwe.yaml
    ├── declipping.yaml
    ├── source_separation.yaml
    ├── unconditional.yaml
    └── vocoding.yaml
```

The main config that contains essential parameters is `inference_cfg.yaml`.

## Inference
To check parameters refer to `configs/inference_cfg.yaml'. For example, inference with diffwave model can be launched as:
```
python main.py model=diffwave task=declipping output_dir="results/declipping_inference" audio_dir="./test_samples/"
```
`audio_dir` is **mandatory** and should be supplied either as path to directory with files for inference for
inverse tasks (bwe, declipping, source_separation, vocoding, any custom one, etc.)
or as **integer** for **unconditional** sampling representing number of audios that will be generated.

## References
- [1] Repository code is adapted from [Improved Denoising Diffusion Probabilistic Models](https://github.com/openai/improved-diffusion).
- [2] Original models were taken and adapted from available open source implementations [Diffwave](https://github.com/lmnt-com/diffwave) and [FFC-AE](https://github.com/SamsungLabs/ffc_se).
- [3] Some inspiration and code for inverse tasks is taken and adapted from [CQTDiff](https://github.com/eloimoliner/CQTdiff).

## Citation

If you find this work useful in your research, please cite:

```
@inproceedings{iashchenko23_interspeech,
  author={Anastasiia Iashchenko and Pavel Andreev and Ivan Shchekotov and Nicholas Babaev and Dmitry Vetrov},
  title={{UnDiff: Unsupervised Voice Restoration with Unconditional Diffusion Model}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={4294--4298},
  doi={10.21437/Interspeech.2023-367}
}
```

Copyright (c) 2023 Samsung Electronics
