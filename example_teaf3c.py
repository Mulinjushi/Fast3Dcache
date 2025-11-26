import os

os.environ['SPCONV_ALGO'] = 'native' 

import imageio
from PIL import Image
from easydict import EasyDict as edict
import torch

from trellis.pipelines import TrellisImageTo3DPipeline_yesfinal
from trellis.pipelines import TrellisImageTo3DPipeline_tea
from trellis.utils import render_utils, postprocessing_utils

total_steps = 25 
anchor_step = 7 


print("Loading pipelines...")
pipeline = TrellisImageTo3DPipeline_tea.from_pretrained("microsoft/TRELLIS-image-large")
pipeline1 = TrellisImageTo3DPipeline_yesfinal.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline1.cuda()
print("Pipelines loaded.")

image_path = "/root/autodl-tmp/Fast3Dcache/assets/example_image/typical_creature_dragon.png"
if not os.path.exists(image_path):
    print(f"Warning: Image path not found: {image_path}")
    print("Using default image path: /root/TRELLIS_REFLOW/assets/example_image/typical_building_building.png")
    image_path = "/root/TRELLIS_REFLOW/assets/example_image/typical_building_building.png"
    
image = Image.open(image_path)
print(f"Loaded image: {image_path}")

f3c_args = edict({
    'use_f3c': True,
    'effective_steps': total_steps,
    'full_sampling_steps': anchor_step,
    'anchor_step': anchor_step,
    'assumed_slope': -0.07,
    'full_sampling_end_steps': int(total_steps * 0.75),
    'aggressive_cache_ratio': 0.7,
    'final_phase_correction_freq': 3,
    'stability_T': 3,
    'ssc_w_A': 0.7,
    'ssc_w_V': 0.3
})

l1_teacache_params = {
    "cache_threshold": 0.1
}

# ---------------------------------------------------------------------
#                A/B   M O D E L   G E N E R A T I O N
# ---------------------------------------------------------------------

print("\n" + "="*50)
print("--- 1. RUNNING: Standard TeaCache (L1 Only) ---")
print("(Stage 1 progress bar is DISABLED)")
print("="*50)

outputs_teacache = pipeline.run(
    image,
    seed=42,
    
    use_teacache_for_structure=True,
    teacache_params=l1_teacache_params,
    
    sparse_structure_sampler_params={
        "steps": total_steps, 
        "cfg_strength": 7.5,
        "cfg_interval": (0.5, 1.0)
    },
    slat_sampler_params={
        "steps": 25,
        "cfg_strength": 3
    },
)
glb_teacache = postprocessing_utils.to_glb(
    outputs_teacache['gaussian'][0],
    outputs_teacache['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb_teacache.export("sample_TEACACHE_L1.glb")

video_teacache = render_utils.render_video(outputs_teacache['mesh'][0])['normal']
imageio.mimsave(f"sample_mesh_TEACACHE_L1_{total_steps}_steps.mp4", video_teacache, fps=30)
print(f"Standard TeaCache (L1) finished. Saved to sample_TEACACHE_L1.glb")

print("\n" + "="*50)
print("--- 2. RUNNING: Nested F3C-TeaCache (L1+L2) ---")
print("(Stage 1 progress bar is DISABLED)")
print("="*50)

outputs_nested = pipeline1.run(
    image,
    seed=42,
    use_f3c_teacache_nested_for_structure=True,
    f3c_teacache_nested_params=l1_teacache_params,
    
    sparse_structure_sampler_params={
        "steps": total_steps,
        "cfg_strength": 7.5,
        "cfg_interval": (0.5, 1.0),
        "f3c_args": f3c_args
    },
    slat_sampler_params={
        "steps": 25,
        "cfg_strength": 3
    },
)
glb_nested = postprocessing_utils.to_glb(
    outputs_nested['gaussian'][0],
    outputs_nested['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb_nested.export("sample_F3C_TC_NESTED.glb")

video_nested = render_utils.render_video(outputs_nested['mesh'][0])['normal']
imageio.mimsave(f"sample_mesh_F3C_TC_NESTED_{total_steps}_steps.mp4", video_nested, fps=30)
print(f"Nested F3C-TeaCache (L1+L2) finished. Saved to sample_F3C_TC_NESTED.glb")

print("\n" + "="*50)
print("All tasks finished. You can now compare:")
print("1. sample_TEACACHE_L1.glb (L1 Only)")
print("2. sample_F3C_TC_NESTED.glb (L1 + L2)")
print("="*50)