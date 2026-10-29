# Adaptive Low-Light Image Enhancement with Self-Paced Learning (A-SPNet+)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An unsupervised, deep learning framework for low-light image enhancement, reproducing and significantly upgrading the **IEEE Transactions on Multimedia (IEEE TMM 2025)** base paper:  
> *"Unsupervised Low-Light Image Enhancement With Self-Paced Learning"* (Yu Luo, Xiangru Chen, Zhihe Lu, Pengyu Li, Yisong Chen).  
> **DOI:** `10.1109/TMM.2024.3521752`

---

## 🌟 Key Highlights & Proposed Innovations (A-SPNet+)

While the baseline SPNet paper achieves competitive results, it suffers from several key limitations:
1. **Rigid Scalar Guidance:** Uses a single global scalar $\lambda$ that treats simple uniform regions and high-contrast edges identically.
2. **Bulky Decoupled Denoiser:** An external 0.74M-parameter denoiser that accounts for >87% of total model size and introduces significant latency.
3. **Local Overexposure:** High Lightness Order Error (LOE = 42.2) causing blown-out highlights in mixed-lighting scenes.

### Our Upgraded Architecture:
* **Spatially-Adaptive SPL Map $\Lambda(x, y)$:** Dynamically assigns pixel-level curriculum guidance weights conditioned on regional darkness and gradient complexity.
* **Integrated Haar Wavelet Denoising:** Decomposes reflectance into high-frequency subbands (LH, HL, HH) with learnable attention gating directly inside the network.
* **SALEM (Spatially-Adaptive Local Exposure Module):** Regulates local illumination intensity to suppress glare and preserve structural contrast.

---

## 📊 Benchmark Evaluation (LOL Dataset - 15 Test Pairs)

| Metric | SPNet (Base Paper Model) | A-SPNet+ (Our Upgraded Model) | Improvement / Advantage |
| :--- | :---: | :---: | :---: |
| **Lightness Order Error (LOE) $\downarrow$** | `42.2` | **`27.7`** | **-34.4% Error Reduction** (100% of test images improved) |
| **Model Parameters $\downarrow$** | `0.85 M` | **`0.56 M`** | **-33.4% More Lightweight** (Mobile/Edge viable) |
| **Inference Latency $\downarrow$** | `21.9 ms` | **`12.3 ms`** | **1.78x Faster** (Real-Time 81+ FPS on RTX GPU) |
| **Reflectance Denoising** | Decoupled CNN | **Wavelet Detail Attention** | Integrated single-stage architecture |
| **Color Fidelity** | Prone to desaturation | **Retinex Chromatic Coupling** | Vibrant, authentic natural color |

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/suryateja2109/low-light-enhancement.git
cd low-light-enhancement
pip install torch torchvision numpy opencv-python Pillow streamlit scipy
```

### 2. Launch the Streamlit Interactive Studio
Launch the real-time Streamlit web studio on `http://localhost:8501`:
```bash
streamlit run streamlit_app.py
```

### 3. Evaluate Both Models
Run automated comparative evaluation across all 15 LOL benchmark pairs:
```bash
python evaluate_all.py
```

### 4. Train Upgraded Model from Scratch
```bash
python train_upgraded.py --epochs 150 --lr 0.0001
```

---

## 📁 Repository Structure
```
├── app.py                 # Flask REST API Studio
├── streamlit_app.py       # Streamlit Real-Time Interactive Studio
├── evaluate_all.py        # Benchmark evaluation script (LOE, PSNR, SSIM, Latency)
├── train_upgraded.py      # Training script for A-SPNet+
├── train_baseline.py      # Baseline SPNet training script
├── train_denoiser.py      # Pre-training script for denoiser module
├── models/
│   ├── aspnet_plus.py     # Our Upgraded A-SPNet+ model
│   ├── spnet_base.py      # IEEE TMM 2025 SPNet baseline
│   ├── fem.py             # Feature Extraction Module
│   ├── lidm.py            # Low-Light Image Decomposition Module
│   └── denoiser.py        # Baseline Denoise Module
├── losses/
│   └── advanced_loss.py   # Ordinal loss, Color loss, Exposure loss
├── weights/
│   ├── aspnet_plus.pth    # Trained weights for A-SPNet+ (2.16 MB)
│   ├── spnet_baseline.pth # Trained weights for SPNet baseline (3.26 MB)
│   └── denoiser_pretrained.pth # Pretrained denoiser weights (1.32 MB)
└── data/
    └── test/LOL/          # 15 LOL benchmark image pairs (low & high)
```

---

## 📜 Citation
```bibtex
@article{luo2025unsupervised,
  title={Unsupervised Low-Light Image Enhancement With Self-Paced Learning},
  author={Luo, Yu and Chen, Xiangru and Lu, Zhihe and Li, Pengyu and Chen, Yisong},
  journal={IEEE Transactions on Multimedia},
  volume={27},
  pages={1808--1821},
  year={2025},
  publisher={IEEE},
  doi={10.1109/TMM.2024.3521752}
}
```

