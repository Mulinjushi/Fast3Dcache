# easycache_baseline/easycache_flow_euler.py

from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from .easycache_manager import EasyCacheManager

class EasyCacheFlowEulerCfgSampler(FlowEulerGuidanceIntervalSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager = None 

    def set_manager(self, manager: EasyCacheManager):
        self.manager = manager

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        if self.manager is None:
            raise ValueError("EasyCache Manager is not set up.")
            
        self.manager.reset()
        
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        
        ret = edict({"samples": None, "pred_x_0": []})
        
        cache_hits = 0
        cache_misses = 0

        v_t_minus_1 = None
        x_t_minus_1 = None

        current_x_t = sample.clone()

        for t_float, t_prev_float in tqdm(t_pairs, desc="EasyCache Sampler", disable=not verbose):
            is_compute_step = self.manager.step_decision(
                current_x_t, 
                v_t_minus_1, 
                x_t_minus_1, 
                total_steps=steps
            )
            
            current_v = None 
            if is_compute_step:
                cache_misses += 1
                t_tensor = torch.tensor([1000 * t_float] * current_x_t.shape[0], device=current_x_t.device)
                
                cond_ = cond if cond.shape[0] == current_x_t.shape[0] else cond.repeat(current_x_t.shape[0], *([1]*(cond.ndim-1)))
                neg_cond_ = neg_cond if neg_cond.shape[0] == current_x_t.shape[0] else neg_cond.repeat(current_x_t.shape[0], *([1]*(neg_cond.ndim-1)))
                
                if cfg_interval[0] <= t_float <= cfg_interval[1]:
                    cond_pred_v = model(current_x_t.float(), t_tensor.float(), cond_.float())
                    uncond_pred_v = model(current_x_t.float(), t_tensor.float(), neg_cond_.float())
                    current_v = uncond_pred_v + cfg_strength * (cond_pred_v - uncond_pred_v)
                else:
                    current_v = model(current_x_t.float(), t_tensor.float(), cond_.float())
                
                current_v = current_v.float()
                
                self.manager.update_cache_after_compute(current_v, current_x_t)
            
            else:
                cache_hits += 1
                current_v = self.manager.get_cached_output(current_x_t) 
            

            v_t_minus_1 = current_v.clone() 
            x_t_minus_1 = current_x_t.clone()

            sample = current_x_t - (t_float - t_prev_float) * current_v.to(sample.dtype)
            current_x_t = sample.clone()
            
            pred_x_0, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t_prev_float, v=current_v.float())
            ret.pred_x_0.append(pred_x_0)
        
        if verbose:
            print(f"[EasyCache Stats] finish {steps} steps")
            print(f"  Cache Hits: {cache_hits} steps")
            print(f"  Cache Misses: {cache_misses} steps")
            if cache_misses > 0:
                print(f"  valid speedup: {steps / cache_misses:.2f}x")

        ret.samples = sample
        return ret