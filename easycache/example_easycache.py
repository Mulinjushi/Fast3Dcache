# /easycache/example_easycache.py
import sys
import os
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from PIL import Image
import torch
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from easycache.easycache_argparser import parse_easycache_args
from easycache.easycache_trellis_pipeline import update_trellis_pipeline_for_easycache

def run_inference_mode(args, pipeline):
    print(f"--- [EasyCache inference] ---")
    image = Image.open(args.image_path)
    sampler_params = {
        "steps": args.effective_steps,
        "cfg_strength": 7.5
    }

    if args.use_easycache:
        print("EasyCache is set up.")
        print(f" Tau: {args.easycache_tau}%")
        print(f" R: {args.easycache_warmup_r} steps")
        pipeline = update_trellis_pipeline_for_easycache(pipeline, args)
    else:
        print("EasyCache is not set up.")

    print(f"Using seed {args.seed} and {args.effective_steps} steps to generate...")
    torch.manual_seed(args.seed)

    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params=sampler_params
    )
    
    if 'gaussian' not in outputs or 'mesh' not in outputs:
       return

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    
    output_path = os.path.join(args.output_dir, f"{args.output_name}.glb")
    
    glb.export(output_path)
    print(f"Inference is finished and saved in {output_path}")

def main():
    args = parse_easycache_args()
    
    print("Loading Trellis pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    pipeline.models['sparse_structure_flow_model'].to(pipeline.device)
    print("Pipeline is loading.")

    run_inference_mode(args, pipeline)

if __name__ == "__main__":
    main()