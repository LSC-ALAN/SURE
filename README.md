# SURE

SURE is a local feature matching model with uncertainty estimation. This repository provides the PyTorch model, Lightning testing/training entrypoints, a single-pair demo.

## Installation

```shell
conda create -n sure python=3.8 -y
conda activate sure

# CUDA 11.8 build. Change this line if your CUDA/PyTorch stack is different.
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install -r requirements.txt
```

## Quick Start

Run SURE on the bundled ScanNet sample pair:

```shell
python runtime_single_pair.py --ckpt_path SURE.ckpt --device cuda
```

The script writes a visualization to `sure_single_pair.png`. If CUDA is not available, use:

```shell
python runtime_single_pair.py --ckpt_path SURE.ckpt --device cpu
```

Use your own pair:

```shell
python runtime_single_pair.py \
  --image0 /path/to/image0.jpg \
  --image1 /path/to/image1.jpg \
  --ckpt_path SURE.ckpt \
  --width 640 \
  --height 480 \
  --device cuda
```

## Testing

Prepare the ScanNet or MegaDepth test subset first, then place or symlink it under `data/{{dataset}}/test`.

```shell
ln -s /path/to/scannet-1500-testset/* data/scannet/test
ln -s /path/to/megadepth-1500-testset/* data/megadepth/test
```

ScanNet:

```shell
bash scripts/reproduce_test/indoor.sh
```

MegaDepth:

```shell
bash scripts/reproduce_test/outdoor.sh
```

Direct Python entry:

```shell
python test.py \
  --data_cfg_path configs/data/scannet_test_1500.py \
  --main_cfg_path configs/sure/indoor/sure_base.py \
  --ckpt_path SURE.ckpt \
  --gpus 1 \
  --num_workers 0
```

## Training

Prepare MegaDepth training data according to your dataset layout, then run:

```shell
bash scripts/reproduce_train/outdoor.sh
```

Direct Python entry:

```shell
python train.py \
  --data_cfg_path configs/data/megadepth_trainval_832.py \
  --main_cfg_path configs/sure/outdoor/sure_base.py \
  --exp_name sure_outdoor \
  --gpus 1
```

## Citation

If you find SURE useful in your research, please consider citing our paper:

```bibtex
@inproceedings{li2026sure,
  author    = {Li, Sicheng and Gu, Zaiwang and Zhang, Jie and Guo, Qing and Jiang, Xudong and Cheng, Jun},
  title     = {{SURE}: Semi-dense {Uncertainty-REfined} Feature Matching},
  booktitle = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  pages     = {10267--10274},
  doi       = {10.1109/ICRA57385.2026.11695914}
}
```

## Acknowledgement

Part of the code is based on EfficientLoFTR and RLE. We thank the authors for their useful source code.
