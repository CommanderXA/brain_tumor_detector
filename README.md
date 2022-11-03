# Brain Tumor Detector

## Description

Project in Deep Learning written in PyTorch

## Data

Data is taken from Kaggle: [Brian Tumor Dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

Create `data` directory inside the `data` directory and place 2 class directories there, renaming them to `healthy` and `tumor`

## Actions

To pass one image for classification run `python3 main.py core.inference=`<*path_to_image*>
To train the model run `python3 main.py core.train=true`
