# Q-OCMNet: Sharpened Quantum Attention Networks for Ocean Color Segmentation

Official implementation of **Q-OCMNet**, a parameter-efficient hybrid quantum-classical neural network for ocean color and coastal plume segmentation on **ISRO Oceansat-3 (EOS-06) OCM-3** multi-spectral satellite data.

---

## 📌 Overview

Satellite-based ocean color monitoring faces severe challenges from atmospheric aerosol interference, sea clutter noise, and large model parameter footprints. 

**Q-OCMNet** integrates a **Sharpened Quantum Attention (SQA)** bottleneck featuring a 4-qubit Parameterized Quantum Circuit (PQC) coupled with hard-sigmoid noise suppression. It achieves state-of-the-art segmentation performance while adding only **918 quantum parameters (+0.86%)** over a classical U-Net backbone.

### Key Performance Highlights
* **Peak Metric (Seed 21):** F1-Score: **0.9842** | mIoU: **0.9689**
* **Aggregated Mean (3 Seeds):** F1-Score: **0.9778 ± 0.0093** | mIoU: **0.9568 ± 0.0178**
* **Model Footprint:** **107,432 parameters** (15.7% fewer parameters than FCN-8s)
