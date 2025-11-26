# example_demo_piliang_case.py

import os
import sys
# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native' 

import imageio
from PIL import Image
from easydict import EasyDict as edict
import torch
import glob
from tqdm import tqdm 


try:
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path)) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if project_root not in sys.path:
         sys.path.insert(0, project_root)

from trellis.pipelines import TrellisImageTo3DPipeline_yesfinal
from trellis.pipelines import TrellisImageTo3DPipeline_tea
from trellis.utils import render_utils, postprocessing_utils

INPUT_DIR = "/yangmengyu/Fast3Dcache_run_demo/assets/example"
OUTPUT_DIR = "./demo_outputs"
SEED = 42

total_steps = 25
anchor_step = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"output is saved in {os.path.abspath(OUTPUT_DIR)}")

print("Loading pipelines (this may take a moment)...")
pipeline = TrellisImageTo3DPipeline_tea.from_pretrained("microsoft/TRELLIS-image-large")
pipeline1 = TrellisImageTo3DPipeline_yesfinal.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline1.cuda()
print("Pipelines loaded and moved to GPU.")

f3c_args = edict({
    'use_f3c': True,
    'effective_steps': total_steps,
    'full_sampling_steps': anchor_step,
    'anchor_step': anchor_step,
    'assumed_slope': -0.07,
    'full_sampling_end_steps': int(total_steps * 0.75),
    'aggressive_cache_ratio': 0.8,
    'final_phase_correction_freq': 3,
    'stability_T': 3,
    'ssc_w_A': 0.7,
    'ssc_w_V': 0.3
})

l1_teacache_params = {
    "cache_threshold": 0.1
}

image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))
if not image_paths:
    print(f"Error!")
    sys.exit(1)

# ---------------------------------------------------------------------
#                 B A T C H   P R O C E S S I N G   L O O P
# ---------------------------------------------------------------------

for image_path in tqdm(image_paths, desc="Processing Images"):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n--- Processing: {base_name}.png ---")
    
    try:
        image = Image.open(image_path)
        print("  Running: 1. Standard TeaCache (L1 Only)...")
        
        outputs_teacache = pipeline.run(
            image,
            seed=SEED,
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

        output_name_l1 = f"{base_name}_teacache.glb"
        output_path_l1 = os.path.join(OUTPUT_DIR, output_name_l1)
        glb_teacache.export(output_path_l1)

        print(f"    Saved: {output_name_l1}")

        print("  Running: 2. Nested F3C-TeaCache (L1+L2)...")

        outputs_nested = pipeline1.run(
            image,
            seed=SEED,
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

        output_name_l1_l2 = f"{base_name}_teacache_f3c.glb"
        output_path_l1_l2 = os.path.join(OUTPUT_DIR, output_name_l1_l2)
        glb_nested.export(output_path_l1_l2)
        print(f"    Saved: {output_name_l1_l2}")

    except Exception as e:
        print(f"Error!")
        import traceback
        traceback.print_exc()
        print("Skip.")

print("\n" + "="*50)
print(f"Finished!")
print("="*50)