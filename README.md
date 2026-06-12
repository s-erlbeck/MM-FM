# MM-FM: Flow Matching for Multimodal Distributions

[![uv](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fuv%2Fmain%2Fassets%2Fbadge%2Fv0.json)](https://github.com/astral-sh/uv)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-mm--fm-yellow)](https://huggingface.co/luo00042/mm-fm)
[![PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **[Flow Matching for Multimodal Distributions](https://mm-flow.github.io/)**  
> [Gaoxiang Luo](https://gaoxiangluo.github.io/)\*, [Frank Cole](https://sites.google.com/umn.edu/frank-cole)\*, Sihang Zhang, Yuxiang Wan, [Yulong Lu](https://lu.math.umn.edu), [Ju Sun](https://sunju.org)  
> _In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026_

[[Project Page](https://mm-flow.github.io/)] [[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Flow_Matching_for_Multimodal_Distributions_CVPR_2026_paper.html)] [[Models](https://huggingface.co/luo00042/mm-fm)] [[BibTeX](#citing-mm-fm)]

## Quickstart

```bash
git clone --recursive https://github.com/GaoxiangLuo/MM-FM.git && cd MM-FM
uv sync                                          # main environment
uv run hf download luo00042/mm-fm --local-dir artifacts  # checkpoints, GMMs, ... (~160 GB)

# generate 50K images with the published mode-conditional + GMM model (8 GPUs)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml \
  --sample-dir results/samples/mode-cond-gmm-8192-diag \
  --num-fid-samples 50000 --precision bf16 --per-proc-batch-size 4

# score them (expected: FID ~ 3.2)
uv run python src/scripts/compute_fid_from_stats.py \
  --ref artifacts/fid_reference/VIRTUAL_imagenet256_labeled.npz \
  --samples results/samples/mode-cond-gmm-8192-diag/*.npz
```

Expected FID for every published checkpoint: see [Results](#results).

## Environment Setup

```bash
uv sync
```

Optional: a TensorFlow environment, only needed for the original
guided-diffusion evaluator (the default PyTorch FID path needs no extra
environment — see [FID Evaluation](#fid-evaluation)):

```bash
uv venv .venv-eval --python 3.11
source .venv-eval/bin/activate
uv pip install 'tensorflow[and-cuda]==2.15.1' numpy==1.23.5 scipy==1.10.0 tqdm==4.67.1
```

## Pretrained Artifacts

All checkpoints, fitted GMMs, RAE (Representation Autoencoder) decoders, and
normalization statistics are
hosted on Hugging Face ([`luo00042/mm-fm`](https://huggingface.co/luo00042/mm-fm)).
Download everything into `artifacts/` at the repository root — the configs
reference these paths directly:

```bash
uv run hf download luo00042/mm-fm --local-dir artifacts
```

To fetch only what one encoder needs (e.g. DINOv2-B, ~72 GB):

```bash
uv run hf download luo00042/mm-fm --local-dir artifacts \
  --include "checkpoints/dinov2-b/*" "checkpoints/autoguidance/*" "gmm/dinov2-b/*" \
            "decoders/dinov2-b/*" "normalization_stats/dinov2-b/*" "fid_reference/*"
```

| Path | Contents |
|---|---|
| `artifacts/checkpoints/<encoder>/` | flow-matching DiT checkpoints (`uncond-gmm` / `mode-gmm`; `25k` = 20 epochs, `100k` = 80 epochs) |
| `artifacts/checkpoints/autoguidance/` | small DiT-S models used as the autoguidance *guide* |
| `artifacts/gmm/<encoder>/` | fitted CLS + spatial GMMs (8192 components, diagonal) |
| `artifacts/decoders/<encoder>/` | pretrained RAE decoders (ViT-XL, from [RAE](https://github.com/bytetriper/RAE)) |
| `artifacts/normalization_stats/<encoder>/` | latent normalization statistics |
| `artifacts/fid_reference/` | ImageNet-256 FID reference batch (mirrored from guided-diffusion) |

Gaussian-baseline checkpoints are not published; train them with the
`*-UNCONDITIONAL.yaml` configs. Pretrained RAE decoders are included
(`artifacts/decoders/`); to train an RAE from scratch, see the upstream
[RAE](https://github.com/bytetriper/RAE) repository. Reproduced FID numbers for every published
checkpoint are listed in [Results](#results).

## ImageNet Data

Training and GMM fitting read the standard **ImageNet-1k (ILSVRC2012) train
set** via `torchvision.datasets.ImageFolder` — i.e. one subfolder per synset
with the original variable-size JPEGs (1,281,167 images, 1,000 classes):

```
/path/to/imagenet/train/
├── n01440764/
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   └── ...
├── n01443537/
│   └── ...
└── ...            # 1,000 synset folders in total
```

No pre-resizing is needed: images are center-cropped to 256×256 on the fly
(ADM-style `center_crop_arr`). Any standard extraction of the ImageNet train
archive into per-class folders works.

## GMM Training

Both GMM settings (unconditional+GMM and mode-conditional+GMM) require a
trained CLS GMM and spatial GMM. Throughout, a **mode** is one component of the
fitted GMM; each image is assigned to the mode its CLS token falls in. **The GMMs used in the paper are included in
the artifact download** (`artifacts/gmm/<encoder>/`), so this step is only
needed to train your own — e.g., for a different encoder or component count.
Works with any of the supported encoders (DINOv2, SigLIP2, MAE). Step 2
instantiates the full RAE, so the [artifact download](#pretrained-artifacts)
(decoder + normalization statistics for your encoder) must be in place.

### Step 1: Fit CLS GMM on ImageNet

Extract CLS tokens (the encoder's global class token, one per image) and fit a
GMM on them. The encoder is automatically loaded from the
config file.

```bash
# With DINOv2 encoder
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/scripts/fit_gmm_imagenet.py \
  --data-path /path/to/imagenet/train \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml \
  --output-dir results/clustering/dinov2-base-imagenet-gmm-8192-diag \
  --n-components 8192 \
  --covariance-type diag \
  --batch-size 64

# With SigLIP2 encoder
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/scripts/fit_gmm_imagenet.py \
  --data-path /path/to/imagenet/train \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL.yaml \
  --output-dir results/clustering/siglip2-base-imagenet-gmm-8192-diag \
  --n-components 8192 \
  --covariance-type diag \
  --batch-size 64

# Output files:
#   - cls_tokens.npz               (cached CLS tokens, reusable)
#   - gmm_n8192_diag_k-means++.pkl (GMM model)
#   - gmm_n8192_diag_k-means++_labels.npy
#   - gmm_n8192_diag_k-means++_weights.png
```

**Reuse cached CLS tokens** for different `n_components` without re-extracting:

```bash
uv run python src/scripts/fit_gmm_imagenet.py \
  --data-path /path/to/imagenet/train \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml \
  --cls-tokens-path results/clustering/dinov2-base-imagenet-gmm-8192-diag/cls_tokens.npz \
  --output-dir results/clustering/dinov2-base-imagenet-gmm-2048-diag \
  --n-components 2048 \
  --covariance-type diag
```

### Step 2: Create Spatial GMM

Transfer cluster assignments from CLS tokens to spatial tokens (the per-patch
encoder latents the flow model is trained on). This produces a GMM suitable for
sampling spatial latent noise during FM training.

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/scripts/spatial_gmm_imagenet.py \
  --cls-gmm-path results/clustering/dinov2-base-imagenet-gmm-8192-diag/gmm_n8192_diag_k-means++.pkl \
  --data-path /path/to/imagenet/train \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml \
  --output-dir results/clustering/dinov2-base-imagenet-gmm-8192-diag \
  --batch-size 32

# Output: spatial_gmm_from_cls_n8192_diag_k-means++_flip.pkl
```

**Resume from Pass 2** (if Pass 1 completed):

```bash
uv run python src/scripts/spatial_gmm_imagenet.py \
  --cls-gmm-path results/clustering/dinov2-base-imagenet-gmm-8192-diag/gmm_n8192_diag_k-means++.pkl \
  --data-path /path/to/imagenet/train \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml \
  --output-dir results/clustering/dinov2-base-imagenet-gmm-8192-diag \
  --skip-pass1
```

### Step 3: Reference the GMM Files in the Training Config

The GMM-setting FM-training configs (`*-GMM-8192-DIAG.yaml`) point at the
fitted GMM files through their `gmm:` block. For the
**mode-conditional + GMM** setting, set `mode_conditional: true`:

```yaml
gmm:
  enabled: true
  mode_conditional: true
  path: 'results/clustering/dinov2-base-imagenet-gmm-8192-diag/spatial_gmm_from_cls_n8192_diag_k-means++_flip.pkl'
  cls_enabled: true
  cls_path: 'results/clustering/dinov2-base-imagenet-gmm-8192-diag/gmm_n8192_diag_k-means++.pkl'
```

For the **unconditional + GMM** setting, set `mode_conditional: false`. The GMM
still provides the source-noise distribution *and* the mode-dependent
data–noise coupling during training (each image is paired with source noise
drawn from its assigned GMM mode); the only difference is that the model itself
is not conditioned on the mode.

## FM Training

Training learns a flow-matching DiT with a decoupled DDT decoder head
(`DiTwDDTHead`; the `DiTDH-XL`/`DiTDH-S` prefixes in config names denote model
size) on encoder latents produced by the frozen RAE. Choose a config for the
desired setting and encoder.

`--data-path` expects the layout described in
[ImageNet Data](#imagenet-data). Training
also reads the RAE decoder, normalization statistics, GMM files, and the FID
reference batch from `artifacts/` (see
[Pretrained Artifacts](#pretrained-artifacts)); to use your own fitted GMMs
instead, update the `gmm:` paths in the config. `--wandb` is optional.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train.py \
  --config <training_config> \
  --data-path /path/to/imagenet/train \
  --results-dir results/flow-matching \
  --precision bf16 \
  --wandb
```

With bf16 precision, it takes approximately 21 training hours on an
8×NVIDIA A100-SXM4-40GB node to reach 20 epochs without evaluation.

### Available training configs (ImageNet-256)

| Setting | DINOv2-B | SigLIP2-B | MAE-B |
|---------|----------|-----------|-------|
| Unconditional | `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml` | `configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL.yaml` | `configs/stage2/training/ImageNet256/DiTDH-XL_MAE-B-UNCONDITIONAL.yaml` |
| Unconditional + GMM | `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` | `configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` | `configs/stage2/training/ImageNet256/DiTDH-XL_MAE-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` |
| Mode-conditional + GMM | `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` | `configs/stage2/training/ImageNet256/DiTDH-XL_SigLIP2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` | `configs/stage2/training/ImageNet256/DiTDH-XL_MAE-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` |

Smaller `DiTDH-S` (DINOv2-B) variants are also available for the unconditional,
unconditional+GMM, and mode-conditional+GMM settings.

The GMM settings use the fitted GMM files from `artifacts/gmm/<encoder>/`
(or your own — see [GMM Training](#gmm-training) above).

Example (mode-conditional + GMM, DINOv2-B):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml \
  --data-path /path/to/imagenet/train \
  --results-dir results/ditxl-mode-cond-gmm-8192-diag \
  --precision bf16 \
  --wandb
```

## Sampling

Sampling configs are self-contained: each embeds the RAE definition, the
checkpoint path (`ckpt`), and (for GMM settings) the GMM file paths. With the
artifact download in place they work **out of the box** — each config points at
the published 80-epoch (`100k`) checkpoint where available. For the 20-epoch
numbers change `ckpt` to the corresponding `*-25k.pt` file (see
[Results](#results)); to sample your own training run, point `ckpt` at
your checkpoint instead.

```bash
# Unconditional
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL.yaml \
  --sample-dir results/samples/uncond \
  --num-fid-samples 50000 \
  --precision bf16 \
  --per-proc-batch-size 4

# Unconditional + GMM
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDH-XL_DINOv2-B-UNCONDITIONAL-GMM-8192-DIAG.yaml \
  --sample-dir results/samples/uncond-gmm-8192-diag \
  --num-fid-samples 50000 \
  --precision bf16 \
  --per-proc-batch-size 4

# Mode-conditional + GMM
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/sample_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml \
  --sample-dir results/samples/mode-cond-gmm-8192-diag \
  --num-fid-samples 50000 \
  --precision bf16 \
  --per-proc-batch-size 4
```

Autoguidance variants (`*-AG.yaml`) of the unconditional+GMM and
mode-conditional+GMM sampling configs are also provided. They guide the main
model with a smaller, less-trained `DiTDH-S` model (guidance scale 1.5). The
published guide models are preconfigured
(`artifacts/checkpoints/autoguidance/`); if you trained your own guide, set
`guidance.guidance_model.ckpt` accordingly.

## FID Evaluation

FID is computed against the official ADM/guided-diffusion ImageNet-256
reference statistics. The reference batch `VIRTUAL_imagenet256_labeled.npz`
(originally from the
[guided-diffusion evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations))
is included in the artifact download at `artifacts/fid_reference/` and embeds
the precomputed full-ImageNet reference moments.

The simplest path runs entirely in the main environment (no TensorFlow). It
uses the same `inception-2015-12-05` network as the TensorFlow evaluator (via
NVIDIA's PyTorch port) and the same embedded reference statistics; it matches
the TensorFlow evaluator to within 0.001 FID on identical samples:

```bash
uv run python src/scripts/compute_fid_from_stats.py \
  --ref artifacts/fid_reference/VIRTUAL_imagenet256_labeled.npz \
  --samples results/samples/mode-cond-gmm-8192-diag/<samples>.npz
```

Alternatively, the original TensorFlow
[guided-diffusion](https://github.com/openai/guided-diffusion) evaluator (the
protocol used for the paper numbers; also reports IS/sFID/precision/recall)
runs in the `.venv-eval` environment (see
[Environment Setup](#environment-setup)); the `guided-diffusion` code is
included as a submodule (clone with `--recursive` or run
`git submodule update --init`):

```bash
CUDA_VISIBLE_DEVICES=0 python guided-diffusion/evaluations/evaluator.py \
  artifacts/fid_reference/VIRTUAL_imagenet256_labeled.npz \
  results/samples/mode-cond-gmm-8192-diag/<samples>.npz
```

## Utility Scripts

### Compute Normalization Statistics (Optional)

Compute position-wise variance statistics for encoder latents. This is
**optional** for encoders that already output normalized features.

```bash
uv run python src/scripts/compute_normalization_stats.py \
  --encoder-cls Dinov2withNorm \
  --encoder-config facebook/dinov2-with-registers-base \
  --data-path /path/to/imagenet/train \
  --output models/stats/dinov2/imagenet/stat.pt \
  --num-samples 50000 \
  --batch-size 64
```

### Compute FID Reference Statistics

Compute InceptionV3 feature statistics for FID evaluation during training.
Only needed for a custom dataset or resolution — for ImageNet-256 the included
`artifacts/fid_reference/VIRTUAL_imagenet256_labeled.npz` is already used by
the training configs.

```bash
# Single GPU
uv run python src/scripts/compute_fid_reference.py \
  --data-path /path/to/imagenet/train \
  --output models/fid_refs/imagenet256.npz \
  --image-size 256 \
  --num-samples 50000

# Multi-GPU (faster)
uv run torchrun --nproc_per_node=4 src/scripts/compute_fid_reference.py \
  --data-path /path/to/imagenet/train \
  --output models/fid_refs/imagenet256.npz \
  --image-size 256 \
  --num-samples 50000
```

## Repository Structure

```
├── configs/              # training + sampling configs (ImageNet-256)
├── src/
│   ├── train.py          # flow-matching DiT training
│   ├── sample_ddp.py     # multi-GPU sampling
│   ├── scripts/          # GMM fitting, FID reference, normalization stats
│   ├── stage1/           # RAE: frozen encoders (DINOv2/SigLIP2/MAE) + ViT decoder
│   └── stage2/           # DiT models + transport (flow matching, GMM sampler)
├── guided-diffusion/     # submodule: FID evaluator
└── artifacts/            # pretrained artifacts from Hugging Face (gitignored)
```

The `stage1`/`stage2` directory names follow the upstream
[RAE](https://github.com/bytetriper/RAE) convention: *stage 1* is the
representation autoencoder (frozen encoder + pretrained decoder, used as-is
here), and *stage 2* is the flow-matching model trained in that representation
space — the part this repository is about.

## Results

FID-50K on ImageNet-256, reproduced end-to-end with this repository's code and the
published artifacts (50-step Euler ODE, bf16 sampling, seed 0; FID via the
guided-diffusion evaluator against `VIRTUAL_imagenet256_labeled.npz`).
AG = autoguidance with the DiT-S guides, scale 1.5.

| Encoder | Setting | Epochs | AG | FID | Checkpoint | Sampling config |
|---|---|---|---|---|---|---|
| DINOv2-B | GMM (uncond) | 20 | -- | 4.84 | `artifacts/checkpoints/dinov2-b/uncond-gmm-25k.pt` | `DiTDH-XL_DINOv2-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` (set `ckpt` to 25k) |
| DINOv2-B | GMM (uncond) | 20 | ✓ | 4.03 | same + `autoguidance/dit-s-uncond-gmm-25k.pt` | `DiTDH-XL_DINOv2-B-UNCONDITIONAL-GMM-8192-DIAG-AG.yaml` (set `ckpt` to 25k) |
| DINOv2-B | GMM (uncond) | 80 | -- | 3.84 | `artifacts/checkpoints/dinov2-b/uncond-gmm-100k.pt` | `DiTDH-XL_DINOv2-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` |
| DINOv2-B | GMM (uncond) | 80 | ✓ | 3.17 | same + `autoguidance/dit-s-uncond-gmm-25k.pt` | `DiTDH-XL_DINOv2-B-UNCONDITIONAL-GMM-8192-DIAG-AG.yaml` |
| DINOv2-B | GMM + Mode | 20 | -- | 4.77 | `artifacts/checkpoints/dinov2-b/mode-gmm-25k.pt` | `DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` (set `ckpt` to 25k) |
| DINOv2-B | GMM + Mode | 20 | ✓ | 4.07 | same + `autoguidance/dit-s-mode-gmm-25k.pt` | `DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG-AG.yaml` (set `ckpt` to 25k) |
| DINOv2-B | GMM + Mode | 80 | -- | 3.20 | `artifacts/checkpoints/dinov2-b/mode-gmm-100k.pt` | `DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` |
| DINOv2-B | GMM + Mode | 80 | ✓ | 2.78 | same + `autoguidance/dit-s-mode-gmm-25k.pt` | `DiTDH-XL_DINOv2-B-MODE-CONDITIONAL-GMM-8192-DIAG-AG.yaml` |
| SigLIP2-B | GMM (uncond) | 20 | -- | 8.23 | `artifacts/checkpoints/siglip2-b/uncond-gmm-25k.pt` | `DiTDH-XL_SigLIP2-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` |
| SigLIP2-B | GMM + Mode | 20 | -- | 7.25 | `artifacts/checkpoints/siglip2-b/mode-gmm-25k.pt` | `DiTDH-XL_SigLIP2-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` |
| MAE-B | GMM (uncond) | 20 | -- | 17.03 | `artifacts/checkpoints/mae-b/uncond-gmm-25k.pt` | `DiTDH-XL_MAE-B-UNCONDITIONAL-GMM-8192-DIAG.yaml` |
| MAE-B | GMM + Mode | 20 | -- | 16.24 | `artifacts/checkpoints/mae-b/mode-gmm-25k.pt` | `DiTDH-XL_MAE-B-MODE-CONDITIONAL-GMM-8192-DIAG.yaml` |

Sampling configs ship pointing at the 80-epoch (100k-step) checkpoints where
available; for 20-epoch numbers, change the `ckpt:` field to the corresponding
`*-25k.pt` file.

## Citing MM-FM

```bibtex
@InProceedings{Luo_2026_CVPR,
  author    = {Luo, Gaoxiang and Cole, Frank and Zhang, Sihang and Wan, Yuxiang and Lu, Yulong and Sun, Ju},
  title     = {Flow Matching for Multimodal Distributions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {23260-23271}
}
```

## Acknowledgments

MM-FM builds directly on [RAE](https://github.com/bytetriper/RAE): we use
RAE's pretrained encoders and decoders as-is and train our flow-matching
models in their latent space; the training/sampling code also
originates from the RAE codebase. FID reference statistics and the TensorFlow
evaluator come from
[guided-diffusion](https://github.com/openai/guided-diffusion).
