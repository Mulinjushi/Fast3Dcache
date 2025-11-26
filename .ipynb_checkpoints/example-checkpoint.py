# example.py
import os
os.environ['ATTN_BACKEND'] = 'xformers'  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'      # Can be 'native' or 'auto', default is 'auto'.
                                        # 'auto' is faster but will do benchmarking at the beginning.
                                        # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open("/root/autodl-tmp/TRELLIS_REFLOW/assets/example_image/typical_building_castle.png")

# --- Baseline: Using the default Euler sampler ---
print("Running with the default Euler sampler...")
outputs_euler = pipeline.run(
    image,
    seed=1,
    sparse_structure_sampler_params={
        "steps": 50,  # Default steps
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 50,
        "cfg_strength": 3,
    },
)
glb = postprocessing_utils.to_glb(
    outputs_euler['gaussian'][0],
    outputs_euler['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample_trellis.glb")

# --- Accelerated: Using the RFLOW sampler ---
# With a 2nd order solver like RFLOW, you can often halve the number of steps 
# and get comparable or even better results.
print("\nRunning with the RFLOW sampler...")
rflow_steps = 23  # For example, half of the original steps
outputs_rflow = pipeline.run(
    image,
    seed=1,
    use_rflow_for_structure=True,
    sparse_structure_sampler_params={
        "steps": rflow_steps,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 50, # Second stage remains the same
        "cfg_strength": 3,
    },
)
glb = postprocessing_utils.to_glb(
    outputs_rflow['gaussian'][0],
    outputs_rflow['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample_REFLOW.glb")

video_rflow = render_utils.render_video(outputs_rflow['mesh'][0])['normal']
imageio.mimsave(f"sample_mesh_rflow_{rflow_steps}_steps.mp4", video_rflow, fps=30)
print("RFLOW sampling finished.")