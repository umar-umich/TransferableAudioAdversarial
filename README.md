# Transferable Adversarial Attacks on Audio Deepfake Detection

[![WACV 2025](https://img.shields.io/badge/WACV-2025-blue)](https://openaccess.thecvf.com/content/WACV2025W/MAPA/papers/Farooq_Transferable_Adversarial_Attacks_on_Audio_Deepfake_Detection_WACVW_2025_paper.pdf)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openaccess.thecvf.com/content/WACV2025W/MAPA/papers/Farooq_Transferable_Adversarial_Attacks_on_Audio_Deepfake_Detection_WACVW_2025_paper.pdf)

> **Official Implementation** | Presented at **WACV 2025 Workshop (MAPA)**

**Authors:** Muhammad Umar Farooq, Awais Khan, Kutub Uddin, Khalid Mahmood Malik  
**Affiliation:** University of Michigan-Flint

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{farooq2025transferable,
  title={Transferable adversarial attacks on audio deepfake detection},
  author={Farooq, Muhammad Umar and Khan, Awais and Uddin, Kutub and Malik, Khalid Mahmood},
  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision},
  pages={1640--1649},
  year={2025}
}
```

---

## Highlights

- **First transferable adversarial attack** for audio deepfake detection preserving transcription and perceptual integrity
- **Significant vulnerabilities exposed** in SOTA systems: accuracy drops from **98% → 26%** (white-box), **92% → 54%** (gray-box), **94% → 84%** (black-box)
- **Novel GAN-based framework** using ensemble surrogate models and Wav2Vec for transcription preservation
- **Comprehensive evaluation** across ASVspoof2019, WaveFake, and In-the-Wild datasets

---

## Overview

This repository introduces a novel GAN-based adversarial attack framework that generates transferable attacks while preserving both transcription integrity and perceptual quality. Unlike traditional methods, our approach uses ensemble surrogate models and self-supervised audio models to create realistic attacks that expose critical vulnerabilities in state-of-the-art audio deepfake detection systems.

---

## Installation

```bash
# Clone repository
git clone https://github.com/umar-umich/TransferableAudioAdversarial.git
cd TransferableAudioAdversarial

# Install dependencies
conda env create -f environment.yml

# Download pretrained models
python scripts/download_models.py
```

**Requirements:** Python 3.9, PyTorch 2.5.1, CUDA 12.6

---

## Datasets

Download and organize the following datasets:

- **ASVspoof2019 LA**: [Official Link](https://datashare.ed.ac.uk/handle/10283/3336)
- **WaveFake**: [GitHub](https://github.com/RUB-SysSec/WaveFake)
- **In-the-Wild**: [Official Link](https://deepfake-total.com/in_the_wild)

---

## Usage

### Training the baseline detection models

```bash
python train_surrogate.py 
```

### Generating Attacks

```bash
python train.py 
```

### Evaluation of Attacks

```bash
python test.py
```

---

## 📊 Results

### Attack Success Rates

| Scenario | ASVspoof2019 | In-the-Wild | WaveFake |
|----------|--------------|-------------|----------|
| **White-box** | 98% → 26% | 95% → 0.4% | 97% → 0.4% |
| **Gray-box** | 92% → 54% | 96% → 64% | 96% → 93% |
| **Black-box** | 95% → 84% | 88% → 64% | 95% → 67% |

### Audio Quality Preservation

| Metric | ASVspoof2019 | In-the-Wild | WaveFake |
|--------|--------------|-------------|----------|
| **PSNR (dB)** | 39.79 | 43.56 | 39.45 |
| **SSIM** | 0.99 | 0.98 | 0.96 |
| **Text Similarity** | 0.95 | 0.87 | 1.00 |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
