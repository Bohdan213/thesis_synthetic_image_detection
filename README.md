## GitHub contains code implementation for thesis on "AI-Generated Image Detection System for Mitigating Fake News and Misinformation"

### Weights:
Pre-trained weights can be found on [Google Drive](https://drive.google.com/drive/folders/1AoOBMky6vzrDjPXAG5IYEVX5xkcnEuRr?usp=sharing).

Folder also contain weights for pre-trained SynCLR model.
Weight were obtained from: https://github.com/google-research/syn-rep-learn/tree/main/SynCLR

### GenImage dataset

[GenImage]( https://genimage-dataset.github.io) was used as main dataset in this thesis for train and evaluation.

Data for training and testing can be found here:
https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS



### Misinfo evaluation dataset

Some state-of-the-art image generation solutions from 2023–2025 are not represented in any open-source datasets. Moreover, the images available in existing datasets do not align with the issue of misinformation and fake news, and don't cover real-case fakes scenarios and sensitive topics. To fill this gap, we develop and introduce evaluation dataset to test our proposed synthetic image detection solutions.

Dataset has folder structure compatible with [GenImage]( https://genimage-dataset.github.io).

Data can be obtained from [Misinfo dataset](https://www.kaggle.com/datasets/bohdan213/misinfo-dataset)


### Code detailes

The repository contains two main notebook files:

- **foundation-classifier.ipynb** — Implementation of a one-branch, global-level solution with a foundation model as the backbone and an MLP head. For this architecture, weights with "foundation_backbone_" in the name should be used.

- **foundation-lbp-classifier.ipynb** — Implementation of a two-branch, global and local-level proposed architecture, with a foundation model as the first branch and an LBP representation followed by ResNet-18 inference as the second branch, with an common MLP head. For this architecture, weights with "foundation_plus_lbp_" in the name should be used.
  
Each file includes code cell with configurable parameters, making it very flexible to use as a script. Simply set up the configuration and run all cells; the training and inference results will be in the last cell.

### Environment

All the experiments were conducted in [Kaggle](https://www.kaggle.com/). No addition set up were required.

If you run this code in a different environment, use `requirements.txt` to install all necessary libraries with the correct versions:

```
pip install -r requirements.txt
```

All experiments were conducted with `CUDA`.