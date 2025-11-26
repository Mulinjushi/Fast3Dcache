# /example_measure_teateaf3c_speed.py
import os

os.environ['SPCONV_ALGO'] = 'native'

import imageio
from PIL import Image
from easydict import EasyDict as edict
import torch
import time 
import numpy as np
from tqdm import tqdm
import glob

from trellis.pipelines import TrellisImageTo3DPipeline_yesfinal
from trellis.pipelines import TrellisImageTo3DPipeline_tea

from trellis.utils import render_utils, postprocessing_utils

BASE_IMAGE_DIR = "/root/autodl-tmp/experiment/TOY4K_transparent1"
total_steps = 25

print("Loading pipeline...")
pipeline = TrellisImageTo3DPipeline_tea.from_pretrained("microsoft/TRELLIS-image-large")
pipeline1 = TrellisImageTo3DPipeline_yesfinal.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline1.cuda()
print("Pipelines loaded.")

image_paths = sorted(glob.glob(os.path.join(BASE_IMAGE_DIR, "*", "*", "0001.png")))

if not image_paths:
    print(f"Error: No images found matching the pattern '{os.path.join(BASE_IMAGE_DIR, '*', '*', '0001.png')}'")
    print("Please check your BASE_IMAGE_DIR path and folder structure.")
    exit()

print(f"Found {len(image_paths)} images to test (e.g., {image_paths[0]}).")
NUM_ROUNDS = len(image_paths) 

anchor_step = 6
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

print("\n" + "="*50)
print(f"--- 1. RUNNING WARM-UP (1 round each) ---")
print("="*50)
try:
    warmup_image = Image.open(image_paths[0])
    warmup_cond_tc = pipeline.get_cond([warmup_image])
    warmup_cond_nested = pipeline1.get_cond([warmup_image])

    _ = pipeline.sample_sparse_structure(
        cond=warmup_cond_tc, num_samples=1, use_teacache=True,
        teacache_params=l1_teacache_params,
        sampler_params={"steps": total_steps, "cfg_strength": 7.5, "cfg_interval": (0.5, 1.0), "verbose": False}
    )

    _ = pipeline1.sample_sparse_structure(
        cond=warmup_cond_nested, num_samples=1, use_f3c_teacache_nested=True,
        f3c_teacache_nested_params=l1_teacache_params,
        sampler_params={"steps": total_steps, "cfg_strength": 7.5, "cfg_interval": (0.5, 1.0), "f3c_args": f3c_args, "verbose": False}
    )
    torch.cuda.synchronize()
    print("Warm-up complete. Starting benchmark...")
except Exception as e:
    print(f"An error occurred during warm-up: {e}")
    print("Skipping benchmark.")
    exit()

# ---------------------------------------------------------------------
#                A/B   B E N C H M A R K   L O O P
# ---------------------------------------------------------------------

print("\n" + "="*50)
print(f"--- 2. RUNNING A/B BENCHMARK ({NUM_ROUNDS} images) ---")
print(f"(Internal progress bars disabled for fair timing)")
print("="*50)

times_tc = []
times_nested = []

for image_path in tqdm(image_paths, desc="Benchmarking Images"):
    try:
        image = Image.open(image_path)
        image_cond_tc = pipeline.get_cond([image])
        image_cond_nested = pipeline1.get_cond([image])
    except Exception as e:
        print(f"Warning: Failed to load or process image {image_path}. Skipping. Error: {e}")
        continue

    torch.cuda.synchronize()
    start_time_tc = time.time()
    
    coords_tc, _ = pipeline.sample_sparse_structure(
        cond=image_cond_tc,
        num_samples=1,
        use_teacache=True,
        teacache_params=l1_teacache_params,
        sampler_params={
            "steps": total_steps,
            "cfg_strength": 7.5,
            "cfg_interval": (0.5, 1.0),
            "verbose": False 
        }
    )
    
    torch.cuda.synchronize()
    end_time_tc = time.time()
    times_tc.append(end_time_tc - start_time_tc)

    torch.cuda.synchronize()
    start_time_nested = time.time()

    coords_nested, _ = pipeline1.sample_sparse_structure(
        cond=image_cond_nested,
        num_samples=1,
        use_f3c_teacache_nested=True,
        f3c_teacache_nested_params=l1_teacache_params,
        sampler_params={
            "steps": total_steps,
            "cfg_strength": 7.5,
            "cfg_interval": (0.5, 1.0),
            "f3c_args": f3c_args,
            "verbose": False 
        }
    )
    
    torch.cuda.synchronize()
    end_time_nested = time.time()
    times_nested.append(end_time_nested - start_time_nested)

# ---------------------------------------------------------------------
#                R E S U L T S
# ---------------------------------------------------------------------

print("\n" + "="*50)
print("--- 3. BENCHMARK RESULTS (Stage 1 Only) ---")

avg_time_tc = np.mean(times_tc)
std_time_tc = np.std(times_tc)
total_time_tc = np.sum(times_tc)
throughput_gen_s_tc = len(times_tc) / total_time_tc
throughput_it_s_tc = (len(times_tc) * total_steps) / total_time_tc

avg_time_nested = np.mean(times_nested)
std_time_nested = np.std(times_nested)
total_time_nested = np.sum(times_nested)
throughput_gen_s_nested = len(times_nested) / total_time_nested
throughput_it_s_nested = (len(times_nested) * total_steps) / total_time_nested


print(f"Total Images Tested:      {len(times_tc)}")
print(f"Sampling Steps per Image: {total_steps}")
print("-"*50)

print(f"[Standard TeaCache (L1)]")
print(f"  Avg. Time / Image:  {avg_time_tc:.4f} s (± {std_time_tc:.4f} s)")
print(f"  Avg. Throughput:    {throughput_gen_s_tc:.2f} Images/s")
print(f"  Avg. Sampler Speed: {throughput_it_s_tc:.2f} it/s (Sampler Steps / sec)")

print(f"\n[Nested F3C-TC (L1+L2)]")
print(f"  Avg. Time / Image:  {avg_time_nested:.4f} s (± {std_time_nested:.4f} s)")
print(f"  Avg. Throughput:    {throughput_gen_s_nested:.2f} Images/s")
print(f"  Avg. Sampler Speed: {throughput_it_s_nested:.2f} it/s (Sampler Steps / sec)")

print("="*50)

if avg_time_nested < avg_time_tc:
    speedup = avg_time_tc / avg_time_nested
    print(f"\n>>> \033[92mSUCCESS: Nested (L1+L2) is {speedup:.2f}x FASTER than L1 only (on average).\033[0m")
else:
    slowdown = avg_time_nested / avg_time_tc
    print(f"\n>>> \033[91mFAILED: Nested (L1+L2) is {slowdown:.2f}x SLOWER than L1 only (on average).\033[0m")
print("="*50)