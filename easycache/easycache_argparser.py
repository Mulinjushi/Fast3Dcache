# easycache_baseline/easycache_argparser.py

import argparse
import math
import os

def parse_easycache_args():
    parser = argparse.ArgumentParser(
        description="Trellis Baseline: EasyCache Acceleration (Zhou et al., 2025)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    mode_group = parser.add_argument_group('Mode Control (EasyCache vs. Baseline)')
    mode_group.add_argument("--use_easycache", action="store_true", 
                            help="--")
    mode_group.add_argument("--euler_steps", type=int, default=25, 
                            help="--")

    easycache_group = parser.add_argument_group('EasyCache Strategy (Zhou et al., 2025)')
    easycache_group.add_argument("--easycache_tau", type=float, default=5.0, 
                                 help="--")
    easycache_group.add_argument("--easycache_warmup_r", type=int, default=5,
                                 help="--")

    io_group = parser.add_argument_group('Trellis I/O & Internal Options')
    io_group.add_argument("--output_dir", type=str, default="outputs_easycache", 
                          help="--")
    io_group.add_argument("--image_path", type=str, default="assets/example_image/typical_building_building.png", 
                          help="--")
    io_group.add_argument("--output_name", type=str, default="sample_easycache", 
                          help="--")
    io_group.add_argument("--seed", type=int, default=42, 
                          help="--")
    
    args = parser.parse_args()
    args.effective_steps = args.euler_steps

    os.makedirs(args.output_dir, exist_ok=True)
            
    return args