# easycache_baseline/easycache_manager.py
import torch

class EasyCacheManager:
    def __init__(self, args):
        print("[EasyCache Manager] initial...")
        self.tau = args.easycache_tau / 100.0
        self.warmup_r = args.easycache_warmup_r
        self.eps = 1e-9 
        
        self.t = 0 
        self.cached_delta = None
        self.cached_k = 1e9
        self.last_computed_v = None
        self.last_computed_x = None
        self.E = 0.0

    def reset(self):
        self.t = 0
        self.cached_delta = None
        self.cached_k = 1e9
        self.last_computed_v = None
        self.last_computed_x = None
        self.E = 0.0

    def step_decision(self, x_t, v_t_minus_1, x_t_minus_1, total_steps) -> bool:
        current_step_index = self.t
        self.t += 1
        
        if current_step_index < self.warmup_r:
            return True 
            
        if current_step_index == total_steps - 1:
            return True

        if v_t_minus_1 is None or x_t_minus_1 is None:
            return True 
            
        norm_v_t_minus_1 = torch.norm(v_t_minus_1, p=1) + self.eps
        norm_x_diff = torch.norm(x_t - x_t_minus_1, p=1)
        
        # k_i * ||x_t - x_{t-1}||
        estimated_v_diff = self.cached_k * norm_x_diff 
        
        # epsilon_t = (estimated_v_diff / ||v_{t-1}||)
        epsilon_t = (estimated_v_diff / norm_v_t_minus_1)
        
        self.E += epsilon_t.item()
        
        if self.E < self.tau:
            return False
        else:
            return True 

    def get_cached_output(self, x_t) -> torch.Tensor:
        """ 
        v_t_hat = x_t + Delta_i 
        """
        return x_t + self.cached_delta

    def update_cache_after_compute(self, v_t, x_t):
        self.cached_delta = v_t - x_t

        if self.last_computed_v is not None:
            norm_v_diff = torch.norm(v_t - self.last_computed_v, p=1) + self.eps
            norm_x_diff = torch.norm(x_t - self.last_computed_x, p=1) + self.eps
            self.cached_k = (norm_v_diff / norm_x_diff).item()
        
        self.last_computed_v = v_t.clone()
        self.last_computed_x = x_t.clone()
        
        self.E = 0.0