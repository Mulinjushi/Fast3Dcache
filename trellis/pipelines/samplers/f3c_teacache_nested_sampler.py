# # trellis/pipelines/samplers/f3c_teacache_nested_sampler.py
# from typing import *
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from tqdm import tqdm
# from easydict import EasyDict as edict

# # 导入 F3C 采样器基类 (来自 f3c_flow_euler.py)
# from fast3Dcache.f3c_flow_euler import F3cFlowEulerCfgSampler
# # 导入 F3C 的 LEADER (来自 f3c_leader.py)
# from fast3Dcache.f3c_leader import LEADER
# from trellis.modules.spatial import unpatchify, patchify

# class F3CTeaCacheNestedSampler(F3cFlowEulerCfgSampler):
#     """
#     "嵌套"融合采样器 (L1-TeaCache + L2-F3C)
    
#     L1 (TeaCache): 决定是否 *完全跳过* 一个计算步骤。
#     L2 (F3C):     如果 L1 决定 *不跳过*，则使用 F3C 的 token-wise 缓存
#                   来减少该步骤的计算量。
#     """

#     def __init__(
#         self,
#         sigma_min: float,
#         cache_threshold: float = 0.1,  # TeaCache (L1) 的缓存阈值
#         **kwargs  # 捕获 F3cFlowEulerCfgSampler 可能需要的其他参数
#     ):
#         # 初始化父类 (F3cFlowEulerCfgSampler)
#         # 注意: sigma_min 必须传递给父类的构造函数
#         super().__init__(sigma_min=sigma_min, **kwargs)
        
#         # L1 TeaCache 参数
#         self.tc_cache_threshold = cache_threshold
#         print(f"F3CTeaCacheNestedSampler initialized:")
#         print(f"  L1 (TeaCache) Threshold (δ): {self.tc_cache_threshold}")
#         print(f"  L2 (F3C) Token-wise cache enabled.")

#     @torch.no_grad()
#     def sample(
#         self, 
#         model, 
#         noise, 
#         cond, 
#         neg_cond, 
#         steps, 
#         cfg_strength, 
#         decoder, # F3C PCSC 锚点步需要解码器
#         args,    # F3C 参数 (来自 LEADER)
#         verbose=True, 
#         cfg_interval: Tuple[float, float] = (0.5, 1.0), 
#         rescale_t: float = 3.0, # 从 f3c_flow_euler.py 移到这里以保持一致性
#         **kwargs
#     ):
#         # --- 1. F3C (L2) 状态初始化 ---
#         # (来自 f3c_flow_euler.py)
#         # LEADER.set_parameters(args) 将在 pipeline 中被调用
#         self._init_f3c_state(noise, args, model) 
        
#         B, C_in, D, H, W = noise.shape
#         # 使用 f3c_leader.py 中的 resolution
#         total_tokens = LEADER.resolution ** 3
        
#         out_patched_channels = C_in 
#         if hasattr(model, 'out_channels') and hasattr(model, 'patch_size') and model.patch_size > 0:
#              # 这是输出通道，F3C的 token 缓存需要这个
#              out_patched_channels = model.out_channels * model.patch_size**3
        
#         sample = noise
#         t_seq = np.linspace(1, 0, steps + 1)
#         # rescale_t (η) [cite: 363]
#         if rescale_t != 1.0:
#             t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
#         t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        
#         ret = edict({"samples": None, "pred_x_0_latents": []})

#         # --- F3C (L2) Token 缓存 ---
#         # (来自 f3c_flow_euler.py)
#         last_pred_v_grid = None # F3C 稳定性追踪器需要 v_t-1 (grid)
#         cached_v_tokens = torch.zeros(B, total_tokens, out_patched_channels, device=noise.device, dtype=torch.float32) # F3C token 缓存 v_t-1 (tokens)

#         # --- TeaCache (L1) 状态初始化 ---
#         # (来自 teacache_sampler.py)
#         cached_output_v = None      # TeaCache 的 *完整* v_t-1 (grid) 缓存
#         proxy_accumulator = 0.0     # TeaCache 代理累加器
#         last_proxy_t_emb = None     # TeaCache 上一个 t_emb 代理
#         t_embedder = model.t_embedder # TeaCache 代理计算器

#         # --- 3. 开始融合循环 ---
#         t_iter = tqdm(
#             t_pairs, 
#             desc="F3C-TC Nested Sampler", 
#             disable=not verbose
#         )
#         for t, t_prev in t_pairs:
            
#             # --- L1: TeaCache 决策 (决定是否跳步) ---
#             is_tc_compute_step = False # 默认：跳步 (Reuse)
            
#             # 1. 计算当前 t_emb 代理
#             t_tensor_tc = torch.tensor([1000 * t] * noise.shape[0], device=noise.device, dtype=torch.float32)
#             current_proxy_t_emb = t_embedder(t_tensor_tc)

#             if last_proxy_t_emb is None:
#                 is_tc_compute_step = True # 第一次迭代，必须计算
#             else:
#                 # 2. 计算代理差异
#                 diff = torch.norm(last_proxy_t_emb - current_proxy_t_emb, p=1)
#                 norm = torch.norm(current_proxy_t_emb, p=1)
#                 rel_l1_diff = diff / (norm + 1e-6)
                
#                 # 3. 累积差异
#                 proxy_accumulator += rel_l1_diff.item()
                
#                 # 4. 决策
#                 if proxy_accumulator > self.tc_cache_threshold:
#                     is_tc_compute_step = True # 超过阈值，必须计算
#                     proxy_accumulator = 0.0   # 重置累加器
            
#             if is_tc_compute_step:
#                 last_proxy_t_emb = current_proxy_t_emb # 仅在计算步更新代理
#             # --- L1 决策结束 ---
            

#             current_v_grid = None
#             final_v_tokens = None # 最终的 v_t (tokens)

#             # --- L2: F3C 计算 或 L1 跳步 ---
#             if is_tc_compute_step:
#                 # 
#                 # === L1: COMPUTE ===
#                 # (执行 L2 F3C 的 Token 级缓存逻辑)
#                 #
#                 t_iter.set_description(f"F3C-TC: Compute L2 (F3C) at t={t:.4f}")
                
#                 # (以下逻辑来自 f3c_flow_euler.py 和 f3c_leader.py)
#                 current_step = LEADER.current_step
                
#                 # F3C 决策：获取本步骤的Token预算 
#                 num_to_skip = LEADER.get_skip_budget_for_current_step(t)
                
#                 is_f3c_active = False
#                 cached_indices, fast_update_indices = None, None
                
#                 # 检查是否激活 F3C (L2)
#                 # 1. 必须在 F3C 阶段 (LEADER 决定)
#                 # 2. 必须有上一步的速度 (F3C SSC 需要)
#                 if num_to_skip > 0 and num_to_skip < total_tokens and last_pred_v_grid is not None:
#                     is_f3c_active = True
#                     # F3C SSC: 选择要更新的 (fast) 和要缓存的 (cached) [cite: 622, 1483]
#                     cached_indices, fast_update_indices = self.stability_tracker.update_and_select(last_pred_v_grid, num_to_skip, t)
#                 else:
#                     # F3C 处于 Phase 1 [cite: 573] 或 Phase 3 校正步 [cite: 755]
#                     # 我们需要计算所有 tokens
#                     fast_update_indices = slice(None) # 等同于 range(0, total_tokens)

#                 # 准备输入
#                 h = patchify(sample.float(), model.patch_size) 
#                 h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
#                 input_tokens_full = model.input_layer(h) + model.pos_emb[None] 
                
#                 # F3C 核心：只选择活跃 Token 
#                 input_tokens = input_tokens_full[:, fast_update_indices, :]
                
#                 t_tensor = torch.tensor([1000 * t] * sample.shape[0], device=sample.device)
                
#                 # F3C CFG 计算 (使用 _run_model_core)
#                 pred_v_tokens = None
#                 cond_ = cond if cond.shape[0] == B else cond.repeat(B, *([1]*(cond.ndim-1)))

#                 if cfg_interval[0] <= t <= cfg_interval[1]:
#                     neg_cond_ = neg_cond if neg_cond.shape[0] == B else neg_cond.repeat(B, *([1]*(neg_cond.ndim-1)))
#                     # F3C：只在活跃 token 上计算
#                     cond_pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)
#                     uncond_pred_v_tokens = self._run_model_core(input_tokens, t_tensor, neg_cond_, model)
#                     pred_v_tokens = uncond_pred_v_tokens + cfg_strength * (cond_pred_v_tokens - uncond_pred_v_tokens)
#                 else:
#                     pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)

#                 # F3C 重组：合并活跃 token 和缓存 token [cite: 707]
#                 if is_f3c_active:
#                     final_v_tokens = cached_v_tokens.clone()
#                     final_v_tokens[:, fast_update_indices, :] = pred_v_tokens
#                 else:
#                     final_v_tokens = pred_v_tokens # 这是一个全量计算

#                 # F3C Unpatch
#                 grid_size = D // model.patch_size
#                 expected_unpatch_shape = (B, out_patched_channels, grid_size, grid_size, grid_size)
#                 reshaped_tokens = final_v_tokens.permute(0, 2, 1).view(*expected_unpatch_shape)
#                 current_v_grid = unpatchify(reshaped_tokens, model.patch_size).contiguous()
                
#                 # PCSC 锚点步逻辑 (来自 f3c_flow_euler.py)
#                 # 检查是否为锚点步 [cite: 591]
#                 if LEADER.current_step == LEADER.anchor_step and not LEADER.schedule_is_set:
#                     # 我们需要 t 和 t-1 的 x0 预测
#                     # 注意：f3c_flow_euler.py 的锚点逻辑是在 *之后* 计算 x0，
#                     # 并且依赖于上一步的 x0。我们在这里调整它。
#                     if len(ret.pred_x_0_latents) > 0:
#                         prev_latent_x0 = ret.pred_x_0_latents[-1] # 获取上一步的 x0
#                         # 计算当前步的 x0
#                         latent_x0_anchor, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t, v=current_v_grid.float())

#                         try:
#                             decoder_device = next(decoder.parameters()).device
#                             # 解码 O_t 和 O_t-1 [cite: 353, 356]
#                             grid_t = (decoder(latent_x0_anchor.to(decoder_device)) > 0) 
#                             grid_t_minus_1 = (decoder(prev_latent_x0.to(decoder_device)) > 0)
#                             # 计算 Δs_t (Eq. 5) [cite: 358]
#                             total_changes = torch.sum(grid_t != grid_t_minus_1).item()
#                             # 记录锚点，计算 PCSC 曲线 [cite: 591, 594]
#                             LEADER.record_complexity_at_anchor(total_changes)
#                             print(f"  F3C Anchor recorded at step {LEADER.current_step}: {total_changes} voxel changes.")
#                         except Exception as e:
#                             print(f"Error during decoder call at anchor step: {e}")
#                     else:
#                         print(f"Warning: Anchor step {LEADER.anchor_step} but no previous x0 latent found.")

                
#             else:
#                 #
#                 # === L1: REUSE ===
#                 # (TeaCache 决定跳过此步)
#                 #
#                 t_iter.set_description(f"F3C-TC: Reuse L1 (TC) at t={t:.4f}")
                
#                 if cached_output_v is None:
#                     raise RuntimeError("F3C-TeaCache logic error: Attempted to reuse L1 cache, but cache is empty.")
                
#                 # 重用 L1 (TeaCache) 的*完整*速度网格
#                 current_v_grid = cached_output_v
#                 # L2 (F3C) 的 *token* 缓存也保持不变
#                 final_v_tokens = cached_v_tokens
            
#             # --- 4. 欧拉更新 (对 L1 和 L2 均适用) ---
#             # [cite: 300]
#             sample = sample - (t - t_prev) * current_v_grid.to(sample.dtype) 
            
#             # --- 5. 状态更新 (为下一次循环准备) ---
            
#             # 更新 L2 (F3C) 状态
#             last_pred_v_grid = current_v_grid.float() # SSC 追踪器需要
#             cached_v_tokens = final_v_tokens.float()  # F3C Token 缓存
            
#             # 更新 L1 (TeaCache) 状态
#             cached_output_v = current_v_grid.float() # L1 (TC) Grid 缓存

#             # 存储 x0 预测 (用于 PCSC 锚点步)
#             latent_x0, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t_prev, v=current_v_grid.float())
#             ret.pred_x_0_latents.append(latent_x0)
            
#             LEADER.increase_step()
        
#         ret.samples = sample
#         return ret

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

# 导入 F3C 采样器基类 (来自 f3c_flow_euler.py)
from fast3Dcache.f3c_flow_euler import F3cFlowEulerCfgSampler
# 导入 F3C 的 LEADER (来自 f3c_leader.py)
from fast3Dcache.f3c_leader import LEADER
from trellis.modules.spatial import unpatchify, patchify

class F3CTeaCacheNestedSampler(F3cFlowEulerCfgSampler):
    """
    "嵌套"融合采样器 (L1-TeaCache + L2-F3C)
    
    L1 (TeaCache): 决定是否 *完全跳过* 一个计算步骤。
    L2 (F3C):     如果 L1 决定 *不跳过*，则使用 F3C 的 token-wise 缓存
                  来减少该步骤的计算量。
    """

    def __init__(
        self,
        sigma_min: float,
        cache_threshold: float = 0.1,  # TeaCache (L1) 的缓存阈值
        **kwargs  # 捕获 F3cFlowEulerCfgSampler 可能需要的其他参数
    ):
        # 初始化父类 (F3cFlowEulerCfgSampler)
        # 注意: sigma_min 必须传递给父类的构造函数
        super().__init__(sigma_min=sigma_min, **kwargs)
        
        # L1 TeaCache 参数
        self.tc_cache_threshold = cache_threshold
        print(f"F3CTeaCacheNestedSampler initialized:")
        print(f"  L1 (TeaCache) Threshold (δ): {self.tc_cache_threshold}")
        print(f"  L2 (F3C) Token-wise cache enabled.")

    @torch.no_grad()
    def sample(
        self, 
        model, 
        noise, 
        cond, 
        neg_cond, 
        steps, 
        cfg_strength, 
        decoder, # F3C PCSC 锚点步需要解码器
        args,    # F3C 参数 (来自 LEADER)
        # verbose=True, # <--- 修复1：已从此行移除
        cfg_interval: Tuple[float, float] = (0.5, 1.0), 
        rescale_t: float = 3.0, 
        **kwargs
    ):
        
        # --- 修复1：从 kwargs 中安全地提取 verbose ---
        verbose = kwargs.get("verbose", True)
        
        # --- 1. F3C (L2) 状态初始化 ---
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

        # --- F3C (L2) Token 缓存 ---
        last_pred_v_grid = None 
        cached_v_tokens = torch.zeros(B, total_tokens, out_patched_channels, device=noise.device, dtype=torch.float32)

        # --- TeaCache (L1) 状态初始化 ---
        cached_output_v = None 
        proxy_accumulator = 0.0
        last_proxy_t_emb = None
        t_embedder = model.t_embedder

        # --- 3. 开始融合循环 ---
        
        # --- 修复2：添加 miniters=1 和 mininterval=0.0 ---
        t_iter = tqdm(
            t_pairs, 
            desc="F3C-TC Nested Sampler", 
            disable=not verbose,
            miniters=1,      # 强制每 1 次迭代就更新一次
            mininterval=0.0  # 强制 0 秒延迟 (立即刷新)
        )
        # -----------------------------------------------

        for t, t_prev in t_iter:
            is_tc_compute_step = False 
            
            t_tensor_tc = torch.tensor([1000 * t] * noise.shape[0], device=noise.device, dtype=torch.float32)
            current_proxy_t_emb = t_embedder(t_tensor_tc)

            if last_proxy_t_emb is None:
                is_tc_compute_step = True
            else:
                diff = torch.norm(last_proxy_t_emb - current_proxy_t_emb, p=1)
                norm = torch.norm(current_proxy_t_emb, p=1)
                rel_l1_diff = diff / (norm + 1e-6)
                
                proxy_accumulator += rel_l1_diff.item()
                
                if proxy_accumulator > self.tc_cache_threshold:
                    is_tc_compute_step = True
                    proxy_accumulator = 0.0
            
            if is_tc_compute_step:
                last_proxy_t_emb = current_proxy_t_emb
            # --- L1 决策结束 ---
            
            current_v_grid = None
            final_v_tokens = None 

            # --- L2: F3C 计算 或 L1 跳步 ---
            if is_tc_compute_step:
                current_step = LEADER.current_step
                num_to_skip = LEADER.get_skip_budget_for_current_step(t)
                
                is_f3c_active = False
                cached_indices, fast_update_indices = None, None
                
                if num_to_skip > 0 and num_to_skip < total_tokens and last_pred_v_grid is not None:
                    is_f3c_active = True
                    # F3C SSC: 选择要更新的 (fast) 和要缓存的 (cached) [cite: 1198, 1150, 1262]
                    cached_indices, fast_update_indices = self.stability_tracker.update_and_select(last_pred_v_grid, num_to_skip, t)
                else:
                    # F3C 处于 Phase 1 [cite: 1160] [cite_start]或 Phase 3 校正步 [cite: 1331]
                    fast_update_indices = slice(None)

                h = patchify(sample.float(), model.patch_size) 
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

                # [cite_start]F3C 重组：合并活跃 token 和缓存 token [cite: 1283]
                if is_f3c_active:
                    final_v_tokens = cached_v_tokens.clone()
                    final_v_tokens[:, fast_update_indices, :] = pred_v_tokens
                else:
                    final_v_tokens = pred_v_tokens

                grid_size = D // model.patch_size
                expected_unpatch_shape = (B, out_patched_channels, grid_size, grid_size, grid_size)
                reshaped_tokens = final_v_tokens.permute(0, 2, 1).view(*expected_unpatch_shape)
                current_v_grid = unpatchify(reshaped_tokens, model.patch_size).contiguous()
                
                # [cite_start]PCSC 锚点步逻辑 [cite: 1166]
                if LEADER.current_step == LEADER.anchor_step and not LEADER.schedule_is_set:
                    if len(ret.pred_x_0_latents) > 0:
                        prev_latent_x0 = ret.pred_x_0_latents[-1]
                        latent_x0_anchor, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t, v=current_v_grid.float())
                        try:
                            decoder_device = next(decoder.parameters()).device
                            # [cite_start]解码 O_t 和 O_t-1 [cite: 929, 932]
                            grid_t = (decoder(latent_x0_anchor.to(decoder_device)) > 0) 
                            grid_t_minus_1 = (decoder(prev_latent_x0.to(decoder_device)) > 0)
                            # [cite_start]计算 Δs_t (Eq. 5) [cite: 934]
                            total_changes = torch.sum(grid_t != grid_t_minus_1).item()
                            # [cite_start]记录锚点，计算 PCSC 曲线 [cite: 1168, 1169]
                            LEADER.record_complexity_at_anchor(total_changes)
                            if verbose: # 仅在 verbose=True 时打印
                                print(f"  F3C Anchor recorded at step {LEADER.current_step}: {total_changes} voxel changes.")
                        except Exception as e:
                            print(f"Error during decoder call at anchor step: {e}")
                    elif verbose:
                        print(f"Warning: Anchor step {LEADER.anchor_step} but no previous x0 latent found.")
                
            else:
                # === L1: REUSE ===
                # (TeaCache 决定跳过此步)
                if cached_output_v is None:
                    raise RuntimeError("F3C-TeaCache logic error: Attempted to reuse L1 cache, but cache is empty.")
                
                current_v_grid = cached_output_v
                final_v_tokens = cached_v_tokens
            
            # --- 4. 欧拉更新 ---
            sample = sample - (t - t_prev) * current_v_grid.to(sample.dtype)
            
            # --- 5. 状态更新 ---
            last_pred_v_grid = current_v_grid.float()
            cached_v_tokens = final_v_tokens.float()
            cached_output_v = current_v_grid.float()

            latent_x0, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t_prev, v=current_v_grid.float())
            ret.pred_x_0_latents.append(latent_x0)
            
            LEADER.increase_step()
        
        ret.samples = sample
        return ret