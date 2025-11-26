# easycache_baseline/easycache_trellis_pipeline.py

from .easycache_flow_euler import EasyCacheFlowEulerCfgSampler
from .easycache_manager import EasyCacheManager

def update_trellis_pipeline_for_easycache(pipeline, args):
    print("Replacing Trellis pipeline as EasyCache sampler...")
    
    original_sampler = pipeline.sparse_structure_sampler
    
    manager = EasyCacheManager(args)
    
    easycache_sampler = EasyCacheFlowEulerCfgSampler(
        sigma_min=original_sampler.sigma_min
    )

    easycache_sampler.set_manager(manager)

    pipeline.sparse_structure_sampler = easycache_sampler
    print(f"GS Stage: {type(original_sampler).__name__} is replaced by {type(easycache_sampler).__name__} (EasyCache).")
    
    return pipeline