# Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/YOUR_ARXIV_ID_HERE)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://mulinjushi.github.io/Fast3Dcache.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Mulinjushi/Fast3Dcache)

AGI Lab,Westlake University
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
  <img src="image/pipeline.png" alt="Fast3Dcache Overview" width="100%">
  </div>

### 1. Predictive Caching Scheduler Constraint (PCSC)
Instead of a fixed caching ratio, PCSC dynamically adjusts the caching budget over timesteps. It leverages the log-linear decay pattern of dynamic voxels to predict how many tokens can be safely cached at each step without harming the geometry structure.

### 2. Spatiotemporal Stability Criterion (SSC)
To determine *which* specific tokens to cache, SSC evaluates voxel stability from two perspectives:
* **Velocity Magnitude:** Reflects the intensity of feature updates.
* **Acceleration Criterion:** Measures the stability of the velocity direction.
By jointly considering these metrics, SSC identifies regions that have converged and can be safely reused.

## 📝 Citation

If you find our work useful for your research, please consider citing:

```bibtex
