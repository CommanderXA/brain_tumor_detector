# Brain Tumor Detector

## Description

Project in Deep Learning written in PyTorch

## Data

Data is taken from Kaggle: [Brian Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

Create `data` directory inside the `data` directory and place 2 class directories there, renaming them to `healthy` and `tumor`

## Actions

To train the model run: `python3 main.py core.train=true`

> *** TensorBoard *** is in this project. To use it run: `tensorboard --logdir=runs` (in the root directory of the project)

To pass one image for classification run: `python3 main.py core.pretrained=<path_to_model> core.inference=<path_to_image>`

> *Pretrained models are available in the releases of this repo.*

## Requierements

- `pip install hydra-core --upgrade`
- `pip install hydra_colorlog --upgrade`
- `pip install pandas`
- `pip install torch torchvision torchaudio`
- `pip install tqdm`
- `pip install tensorboard`
- `pip install torch_tb_profiler`
