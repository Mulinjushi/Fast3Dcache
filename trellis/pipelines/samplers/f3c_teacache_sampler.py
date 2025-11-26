# trellis/pipelines/samplers/f3c_teacache_sampler.py
from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from .teacache_sampler import TeaCacheSampler

class F3CTeaCacheSampler(TeaCacheSampler):
    def __init__(
        self,
        sigma_min: float,
        tc_base_threshold: float = 0.1,
        pcsc_rho_a: float = 0.1,
        pcsc_mu: float = -0.07,
        pcsc_sigma: float = 30000,
        pcsc_patch_size: int = 2,
        pcsc_threshold_scale: float = 2.0
    ):
        super().__init__(sigma_min=sigma_min, cache_threshold=tc_base_threshold)
        self.tc_base_threshold = tc_base_threshold
        self.pcsc_rho_a = pcsc_rho_a
        self.pcsc_mu = pcsc_mu
        self.pcsc_sigma = pcsc_sigma
        self.pcsc_patch_size = pcsc_patch_size
        self.pcsc_threshold_scale = pcsc_threshold_scale
        
        print(f"F3CTeaCacheSampler (Fixed) initialized:")
        print(f"  TC Base Threshold (δ_base): {self.tc_base_threshold}")
        print(f"  F3C PCSC Params (rho_a={pcsc_rho_a}, mu={pcsc_mu}, sigma={pcsc_sigma})")

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        neg_cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 3.0,
        cfg_strength: float = 7.5,
        cfg_interval: Tuple[float, float] = (0.5, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        cached_output_v = None
        proxy_accumulator = 0.0
        last_proxy_t_emb = None 
        t_embedder = model.t_embedder

        k_anchor = int(steps * self.pcsc_rho_a)
        try:
            N_p_latent = (model.resolution // self.pcsc_patch_size) ** 3
        except AttributeError:
            N_p_latent = 4096
        
        t_iter = tqdm(t_pairs, desc="F3C-TeaCache Sampling", disable=not verbose)
        for k, (t, t_prev) in enumerate(t_iter):
            k_idx = k + 1
            current_threshold = self.tc_base_threshold
            if k_idx <= k_anchor:
                t_iter.set_description(f"F3C Phase 1 (k={k_idx}): Using TC Base Thresh={current_threshold:.4f}")
            else:
                delta_s_hat = self.pcsc_sigma * np.exp(self.pcsc_mu * (k_idx - k_anchor))
            
                c_t_latent = N_p_latent - delta_s_hat 
                
                cache_ratio = np.clip(c_t_latent / N_p_latent, 0, 1)
            
                dynamic_bonus = self.tc_base_threshold * (self.pcsc_threshold_scale * cache_ratio)
                current_threshold = self.tc_base_threshold + dynamic_bonus
                
                t_iter.set_description(f"F3C Phase 2/3 (k={k_idx}): Ratio={cache_ratio:.2f} | DynThresh={current_threshold:.4f}")

            is_cache_step = False
            
            t_tensor = torch.tensor([1000 * t] * noise.shape[0], device=noise.device, dtype=torch.float32)
            current_proxy_t_emb = t_embedder(t_tensor)

            if last_proxy_t_emb is None:
                is_cache_step = True
            else:
                diff = torch.norm(last_proxy_t_emb - current_proxy_t_emb, p=1)
                norm = torch.norm(current_proxy_t_emb, p=1)
                rel_l1_diff = diff / (norm + 1e-6)
                
                proxy_accumulator += rel_l1_diff.item()
                
                if proxy_accumulator <= current_threshold:
                    is_cache_step = False
                else:
                    is_cache_step = True
                    proxy_accumulator = 0.0

            if is_cache_step:
                last_proxy_t_emb = current_proxy_t_emb

            if is_cache_step:

                if t >= cfg_interval[0] and t <= cfg_interval[1]:
                    pred_x_0_unc, _, pred_v_unc = self._get_model_prediction(model, sample, t, neg_cond, **kwargs)
                    pred_x_0_c, _, pred_v_c = self._get_model_prediction(model, sample, t, cond, **kwargs)
                    
                    pred_v = (1 + cfg_strength) * pred_v_c - cfg_strength * pred_v_unc
                    pred_x_0 = (1 + cfg_strength) * pred_x_0_c - cfg_strength * pred_x_0_unc
                else:
                    pred_x_0, _, pred_v = self._get_model_prediction(model, sample, t, cond, **kwargs)
                
                cached_output_v = pred_v
            
            else:
                
                if cached_output_v is None:
                    raise RuntimeError("F3C-TeaCache logic error: Attempted to reuse cache, but cache is empty.")
                
                pred_v = cached_output_v
                pred_x_0, _ = self._v_to_xstart_eps(sample, t, pred_v)
            
            pred_x_prev = sample - (t - t_prev) * pred_v
            sample = pred_x_prev
            
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(pred_x_0)
        
        ret.samples = sample
        return ret