# Contrastive Learning with Counterfactual Explanations for Radiology Report Generation

This repository contains the official implementation of the paper **[Contrastive Learning with Counterfactual Explanations for Radiology Report Generation](https://link.springer.com/chapter/10.1007/978-3-031-72775-7_10)**, presented at ECCV 2024. The paper introduces a novel approach to improve radiology report generation by leveraging contrastive learning with counterfactual explanations.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Annotation Files](#annotation-files)
- [Code Structure](#code-structure)
- [Counterfactual Generation Process](#counterfactual-generation-process)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview
This code is built upon existing frameworks, including **[R2Gen](https://github.com/cuhksz-nlp/R2Gen)**, **[DCL](https://github.com/mlii0117/DCL)**, and **[BLIP](https://github.com/salesforce/BLIP)**. It introduces contrastive learning mechanisms that incorporate counterfactual explanations, improving both the interpretability and quality of radiology report generation.

Key features:
- Contrastive learning for robust feature learning.
- Counterfactual explanations to highlight the contribution of critical features in report generation.

## Requirements
Install the dependencies using the `requirements.txt` file provided:
```bash
pip install -r requirements.txt
```

## Annotation Files

The annotation files required for training and evaluation can be downloaded from our **[DCL](https://github.com/mlii0117/DCL)** project. Ensure the annotation files are placed in the appropriate directories as specified in the code.


## Counterfactual Generation Process

The implementation for counterfactual generation can be found in the file [models/blip.py at line 419](https://github.com/mlii0117/CoFE/blob/main/models/blip.py#L419). This process is a crucial component of our contrastive learning framework, enabling the generation of counterfactual explanations during training.

## Citation

If you find this code useful for your research, please consider citing our work:

```bash
@inproceedings{li2025contrastive,
  title={Contrastive learning with counterfactual explanations for radiology report generation},
  author={Li, Mingjie and Lin, Haokun and Qiu, Liang and Liang, Xiaodan and Chen, Ling and Elsaddik, Abdulmotaleb and Chang, Xiaojun},
  booktitle={European Conference on Computer Vision},
  pages={162--180},
  year={2025},
  organization={Springer}
}
```

Acknowledgements

We would like to thank the authors of R2Gen, DCL, and BLIP for making their code publicly available, which served as the foundation for this project.
