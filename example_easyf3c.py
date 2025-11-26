# example_easyf3c.py

import sys
import os
os.environ['SPCONV_ALGO'] = 'native'

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.dirname(project_root)) 

import torch
import time
from PIL import Image
from easydict import EasyDict as edict
from fast3Dcache.f3c_argparser import parse_f3c_args
from trellis.pipelines.trellis_image_to_3d_easyf3c import TrellisImageTo3DEasyF3CPipeline
from trellis.utils import postprocessing_utils

def main():
    args = parse_f3c_args()
    if not args.use_f3c:
        args.use_f3c = True
        args.full_sampling_steps = max(1, int(args.effective_steps * args.full_sampling_ratio))
        args.full_sampling_end_steps = max(args.full_sampling_steps + 1, int(args.effective_steps * args.full_sampling_end_ratio))
        args.anchor_step = max(1, int(args.effective_steps * args.anchor_ratio))

    easy_f3c_nested_params = {
        "ec_tau": 50,
        "ec_warmup_R": 2
    }
    
    print("\n" + "="*50)
    print("EasyCache (L1) + Fast3Dcache (L2) test")
    print(f"L1 (EasyCache) τ: {easy_f3c_nested_params['ec_tau']}%")
    print(f"L1 (EasyCache) R: {easy_f3c_nested_params['ec_warmup_R']} steps")
    print(f"L2 (F3C) Anchor Step: {args.anchor_step} steps")
    print(f"L2 (F3C) Total Steps: {args.effective_steps} steps")
    print("="*50)

    print("EasyF3C pipeline Loading")
    pipeline = TrellisImageTo3DEasyF3CPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    image = Image.open(args.image_path)
    
    f3c_args_easydict = edict(vars(args))
    
    sampler_params = {
        "steps": args.effective_steps,
        "cfg_strength": 7.5,
        "cfg_interval": (0.5, 1.0),
        "f3c_args": f3c_args_easydict
    }

    print(f"Using seed {args.seed} and {args.effective_steps} steps to generate...")
    
    start_time = time.time()
    
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params=sampler_params,
        use_easy_f3c_nested_for_structure=True,
        easy_f3c_nested_params=easy_f3c_nested_params
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"\n--- (cost: {end_time - start_time:.4f} s) ---")

    
    if 'gaussian' not in outputs or 'mesh' not in outputs:
         print("\n ERROR!")
         return

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}_easyf3c.glb")
    
    glb.export(output_path)
    print(f"Finished! Output is saved in {output_path}")

if __name__ == "__main__":
    main()