# Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/YOUR_ARXIV_ID_HERE)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://YOUR_PROJECT_PAGE_URL)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_SPACE_URL)

**Mengyu Yang**, **Yanming Yang**, **Tong Zhao**, **Chenyi Xu**, **Ruibo Li**, **Chenxi Song**, **Chi Zhang**$^*$, **Yufan Zuo**

*Westlake University, Hangzhou Dianzi University, UESTC, Renmin University of China, NTU*

</div>

---

## 📖 Abstract

Diffusion models have achieved impressive generative quality across modalities like 3D shapes, but their inference remains computationally expensive due to the iterative denoising process. While recent caching-based methods effectively reuse redundant computations for 2D/Video, directly applying them to 3D diffusion models can disrupt geometric consistency.

To overcome this, we propose **Fast3Dcache**, a training-free geometry-aware caching framework that accelerates 3D diffusion inference while preserving geometric fidelity. Our method introduces:
1.  **Predictive Caching Scheduler Constraint (PCSC)**: Dynamically determines cache quotas according to voxel stabilization patterns.
2.  **Spatiotemporal Stability Criterion (SSC)**: Selects stable features for reuse based on velocity magnitude and acceleration criteria.

Comprehensive experiments show that Fast3Dcache accelerates inference significantly (up to **27.12% speed-up** and **54.8% FLOPs reduction** on TRELLIS) with minimal degradation in geometric quality.

## 🚀 Method

Our approach is motivated by the observation of a **Three-Phase Stabilization Pattern** in voxel occupancy during the denoising process.

<div align="center">
  <img src="assets/overview.png" alt="Fast3Dcache Overview" width="100%">
  </div>

### 1. Predictive Caching Scheduler Constraint (PCSC)
Instead of a fixed caching ratio, PCSC dynamically adjusts the caching budget over timesteps. It leverages the log-linear decay pattern of dynamic voxels to predict how many tokens can be safely cached at each step without harming the geometry structure.

### 2. Spatiotemporal Stability Criterion (SSC)
To determine *which* specific tokens to cache, SSC evaluates voxel stability from two perspectives:
* **Velocity Magnitude:** Reflects the intensity of feature updates.
* **Acceleration:** Measures the stability of the velocity direction.
By jointly considering these metrics, SSC identifies regions that have converged and can be safely reused.

## 📊 Performance

Fast3Dcache achieves state-of-the-art acceleration-performance trade-offs.

| Method | Throughput (iters/s) ↑ | FLOPs (T) ↓ | CD ↓ | F-Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| TRELLIS (Vanilla) | 0.5055 | 244.2 | 0.0686 | 54.82 |
| **Fast3Dcache ($\tau=3$)** | **0.5850** | **142.4** | **0.0697** | **54.09** |
| **Fast3Dcache ($\tau=8$)** | **0.6426** | **110.3** | **0.0703** | **53.75** |

> Fast3Dcache can also be combined with **TeaCache** and **EasyCache** for even greater speedups (up to **10.33x** acceleration with EasyCache + Ours).

## 📝 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{yang2025fast3dcache,
  title={Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration},
  author={Yang, Mengyu and Yang, Yanming and Zhao, Tong and Xu, Chenyi and Li, Ruibo and Song, Chenxi and Zhang, Chi and Zuo, Yufan},
  journal={arXiv preprint arXiv:24xx.xxxxx},
  year={2025}
}