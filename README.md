# Enhancing Radiology Report Generation via Multi-Phased Supervision

## Introduction
![overview](https://github.com/zailongchen/MultiP-R2Gen/blob/main/images/frame.png?raw=true)

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
https://github.com/zailongchen/MultiP-R2Gen.git
cd MultiP-R2Gen
pip install -r requirements.txt
```


**2. Prepare the training dataset**

IU-xray: you can download our preprocess annotation files from [label](https://drive.google.com/file/d/1xVGXTeRw9WedqNbKOBPJwN4EgsQXbkP6/view?usp=sharing), [triple](https://drive.google.com/file/d/1BZyPmoiretALyvzevdRMOD6nTXKD3o0r/view?usp=sharing), and [report](https://drive.google.com/file/d/1G7l3J769SvYDCzKemHUDi5ta4Lq7Vvra/view?usp=sharing). And download the dataset from [here](https://drive.google.com/file/d/1RBD2nZemEt6lYdSHBL5Uzq4t0Gz79oDq/view?usp=sharing)

Mimic-cxr: you can download our preprocess annotation files from [label](https://drive.google.com/file/d/15-e0RZ_Rn4PiNI0WaYIlvTbVOF_IbdBs/view?usp=sharing), [triple](https://drive.google.com/file/d/1etucWFKKHIX6RIt6T3hvIrXl8CBCMPsN/view?usp=sharing), and [report](https://drive.google.com/file/d/1qVozZw_isNLDIXTp1Uaz32J21b1lQxjD/view?usp=sharing). And download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training

For IU-Xray

Phase 1: Label Prediction

```bash
bash scripts/iuxray/1-label_train.sh
```

Phase 2: Triple Extraction

```bash
bash scripts/iuxray/2-triple_train.sh
```

Phase 3: Report Generation

```bash
bash scripts/iuxray/3-report_train.sh
```

For MIMIC-CXR

Phase 1: Label Prediction

```bash
bash scripts/mimic/1-label_train.sh
```

Phase 2: Triple Extraction

```bash
bash scripts/mimic/2-triple_train.sh
```

Phase 3: Report Generation

```bash
bash scripts/mimic/3-report_train.sh
```

### Testing

For IU-Xray

```bash
bash scripts/iuxray/3-report_test.sh
```

For MIMIC-CXR

```bash
bash scripts/mimic/3-report_test.sh
```

## Acknowledgement

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) This repo is mainly built upon R2GenGPT. We sincerely appreciate the authors' contributions to the original implementation.
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
