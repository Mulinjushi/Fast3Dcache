# trellis/pipelines/samplers/easy_f3c_nested_sampler.py

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from fast3Dcache.f3c_flow_euler import F3cFlowEulerCfgSampler
from fast3Dcache.f3c_leader import LEADER
from trellis.modules.spatial import unpatchify, patchify

class EasyF3CNestedSampler(F3cFlowEulerCfgSampler):
    def __init__(
        self,
        sigma_min: float,
        ec_tau: float = 5.0,
        ec_warmup_R: int = 10,
        **kwargs
    ):
        super().__init__(sigma_min=sigma_min, **kwargs)
        self.ec_tau = ec_tau / 100.0
        self.ec_warmup_R = ec_warmup_R
        
        print(f"EasyF3CNestedSampler (L1+L2) Initialized:")
        print(f"  L1 (EasyCache) τ (Tau): {self.ec_tau * 100.0:.1f}% [cite: 1902]")
        print(f"  L1 (EasyCache) R (Warmup): {self.ec_warmup_R} steps [cite: 1911]")
        print(f"  L2 (Fast3Dcache) Token-wise cache enabled.")

    @torch.no_grad()
    def sample(
        self, 
        model, 
        noise, 
        cond, 
        neg_cond, 
        steps, 
        cfg_strength, 
        decoder,
        args,
        cfg_interval: Tuple[float, float] = (0.5, 1.0), 
        rescale_t: float = 3.0, 
        **kwargs
    ):
        
        verbose = kwargs.get("verbose", True)
        
        self._init_f3c_state(noise, args, model)
        
        B, C_in, D, H, W = noise.shape
        total_tokens = LEADER.resolution ** 3
        
        out_patched_channels = C_in 
        if hasattr(model, 'out_channels') and hasattr(model, 'patch_size') and model.patch_size > 0:
            out_patched_channels = model.out_channels * model.patch_size**3
        
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        if rescale_t != 1.0:
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        
        ret = edict({"samples": None, "pred_x_0_latents": []})

        last_pred_v_grid = None 
        cached_v_tokens = torch.zeros(B, total_tokens, out_patched_channels, device=noise.device, dtype=torch.float32)
        l1_E_t = 0.0          
        l1_Delta_i = None        
        l1_k_i = 1.0               
        l1_x_i = None            
        l1_v_i = None
        
        l1_last_x_t_for_eps = None
        l1_last_v_t_for_eps = None
        
        eps = 1e-6
        t_iter = tqdm(
            t_pairs, 
            desc="EasyF3C Nested Sampler", 
            disable=not verbose,
            miniters=1,
            mininterval=0.0
        )
        
        for t, t_prev in t_iter:
            
            current_x_t_float = sample.float()
            is_l1_compute_step = False
            
            current_step_index = LEADER.current_step

            if current_step_index < self.ec_warmup_R:
                is_l1_compute_step = True
                t_iter.set_description(f"EasyF3C: L1 Warmup (Step {current_step_index+1}/{self.ec_warmup_R})")
            
            elif l1_Delta_i is None:
                is_l1_compute_step = True
            
            else:
                if l1_last_x_t_for_eps is None or l1_last_v_t_for_eps is None:
                    is_l1_compute_step = True
                else:
                    delta_x_norm = torch.norm(current_x_t_float - l1_last_x_t_for_eps, p=1)
                    last_v_norm = torch.norm(l1_last_v_t_for_eps, p=1)
                    
                    epsilon_n = (l1_k_i * delta_x_norm) / (last_v_norm + eps)
                    
                    l1_E_t += epsilon_n.item()
                    
                    if l1_E_t >= self.ec_tau:
                        is_l1_compute_step = True
                    else:
                        is_l1_compute_step = False

            current_v_grid = None
            final_v_tokens = None

            if is_l1_compute_step:
                num_to_skip = LEADER.get_skip_budget_for_current_step(t)
                
                is_f3c_active = False
                cached_indices, fast_update_indices = None, None

                if num_to_skip > 0 and num_to_skip < total_tokens and last_pred_v_grid is not None:
                    is_f3c_active = True
                    cached_indices, fast_update_indices = self.stability_tracker.update_and_select(last_pred_v_grid, num_to_skip, t)
                    t_iter.set_description(f"EasyF3C: Compute (L2 Active: {len(fast_update_indices)}/{total_tokens} tokens)")
                else:
                    t_iter.set_description(f"EasyF3C: Compute (L2 Full Step)")
                    fast_update_indices = slice(None)

                h = patchify(current_x_t_float, model.patch_size) 
                h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
                input_tokens_full = model.input_layer(h) + model.pos_emb[None] 

                input_tokens = input_tokens_full[:, fast_update_indices, :]
                
                t_tensor = torch.tensor([1000 * t] * sample.shape[0], device=sample.device)

                pred_v_tokens = None
                cond_ = cond if cond.shape[0] == B else cond.repeat(B, *([1]*(cond.ndim-1)))

                if cfg_interval[0] <= t <= cfg_interval[1]:
                    neg_cond_ = neg_cond if neg_cond.shape[0] == B else neg_cond.repeat(B, *([1]*(neg_cond.ndim-1)))
                    cond_pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)
                    uncond_pred_v_tokens = self._run_model_core(input_tokens, t_tensor, neg_cond_, model)
                    pred_v_tokens = uncond_pred_v_tokens + cfg_strength * (cond_pred_v_tokens - uncond_pred_v_tokens)
                else:
                    pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)

                if is_f3c_active:
                    final_v_tokens = cached_v_tokens.clone()
                    final_v_tokens[:, fast_update_indices, :] = pred_v_tokens
                else:
                    final_v_tokens = pred_v_tokens

                grid_size = D // model.patch_size
                expected_unpatch_shape = (B, out_patched_channels, grid_size, grid_size, grid_size)
                reshaped_tokens = final_v_tokens.permute(0, 2, 1).view(*expected_unpatch_shape)
                current_v_grid = unpatchify(reshaped_tokens, model.patch_size).contiguous()
                
                current_v_grid_float = current_v_grid.float()
                
                if current_step_index == LEADER.anchor_step and not LEADER.schedule_is_set:
                    if len(ret.pred_x_0_latents) > 0:
                        prev_latent_x0 = ret.pred_x_0_latents[-1]
                        latent_x0_anchor, _ = self._v_to_xstart_eps(x_t=current_x_t_float, t=t, v=current_v_grid_float)

                        try:
                            decoder_device = next(decoder.parameters()).device
                            grid_t = (decoder(latent_x0_anchor.to(decoder_device)) > 0) 
                            grid_t_minus_1 = (decoder(prev_latent_x0.to(decoder_device)) > 0)
                            total_changes = torch.sum(grid_t != grid_t_minus_1).item()
                            LEADER.record_complexity_at_anchor(total_changes)
                            if verbose:
                                print(f"  L2 (F3C) Anchor recorded at step {current_step_index}: {total_changes} voxel changes.")
                        except Exception as e:
                            print(f"Error during decoder call at L2 anchor step: {e}")
                    elif verbose:
                        print(f"Warning: L2 Anchor step {LEADER.anchor_step} but no previous x0 latent found.")

                l1_E_t = 0.0
                l1_Delta_i = current_v_grid_float - current_x_t_float 
                if l1_v_i is not None and l1_x_i is not None:
                    k_delta_v_norm = torch.norm(current_v_grid_float - l1_v_i, p=1)
                    k_delta_x_norm = torch.norm(current_x_t_float - l1_x_i, p=1)
                    l1_k_i = k_delta_v_norm / (k_delta_x_norm + eps)

                l1_x_i = current_x_t_float.clone()
                l1_v_i = current_v_grid_float.clone()
                
            else:
                t_iter.set_description(f"EasyF3C: Reuse L1 (E_t={l1_E_t*100:.1f}%)")
                
                if l1_Delta_i is None:
                    raise RuntimeError("EasyF3C logic error: Attempted to reuse L1 cache, but cache (Delta_i) is empty.")

                current_v_grid_float = current_x_t_float + l1_Delta_i
                current_v_grid = current_v_grid_float.to(sample.dtype)

                final_v_tokens = cached_v_tokens

            pred_x_prev = sample - (t - t_prev) * current_v_grid.to(sample.dtype)
            sample = pred_x_prev

            last_pred_v_grid = current_v_grid.float()
            cached_v_tokens = final_v_tokens.float()
            
            l1_last_x_t_for_eps = sample.float().clone()
            l1_last_v_t_for_eps = current_v_grid.float().clone() 

            latent_x0, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t_prev, v=current_v_grid.float())
            ret.pred_x_0_latents.append(latent_x0)

            LEADER.increase_step()
        
        ret.samples = sample
        return ret