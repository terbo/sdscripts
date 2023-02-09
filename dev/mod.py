''' mod.py v.1
taken from mixture-of-diffusion/generate_grid_from_json.py
this sample json works on 8GB, but any higher resolution will need more VRAM!

sample json:
{
    "cpu_vae": true,
    "model": "prompthero/openjourney-v2",
    "gc": 8,
    "gc_tiles": null,
    "prompt": [
        [
                "A calm sea bed in the depths of the ocean, by jakub rozalski, sun rays, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                "A giant rock covered in alge, by jakub rozalski, sun rays, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
                "A large florescent jelly fish swimming,by jakub rozalski, sun rays, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
        ]
    ],
    "scheduler": "lms",
    "seed": "random",
    "steps": 50,
    "tile_col_overlap": 256,
    "tile_height": 512,
    "tile_row_overlap": 256,
    "tile_width": 400
}

'''

import argparse
import datetime
from diffusers import LMSDiscreteScheduler, DDIMScheduler
import json
import torch
import np

from diffusiontools.tiling import StableDiffusionTilingPipeline

def generate_grid(generation_arguments):
    model_id = generation_arguments["model"]
    # Prepared scheduler
    if generation_arguments["scheduler"] == "ddim":
        scheduler = DDIMScheduler()
    elif generation_arguments["scheduler"] == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    else:
        raise ValueError(f"Unrecognized scheduler {generation_arguments['scheduler']}")
    pipe = StableDiffusionTilingPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to("cuda:0")

    if generation_arguments['seed'] == "random":
       generation_arguments['seed'] = np.random.randint(99999999)

    pipeargs = {
        "guidance_scale": generation_arguments["gc"],
        "num_inference_steps": generation_arguments["steps"],
        "seed": generation_arguments["seed"],
        "prompt": generation_arguments["prompt"],
        "tile_height": generation_arguments["tile_height"], 
        "tile_width": generation_arguments["tile_width"], 
        "tile_row_overlap": generation_arguments["tile_row_overlap"], 
        "tile_col_overlap": generation_arguments["tile_col_overlap"],
        "guidance_scale_tiles": generation_arguments["gc_tiles"],
        "cpu_vae": generation_arguments["cpu_vae"] if "cpu_vae" in generation_arguments else False,
    }
    if "seed_tiles" in generation_arguments: pipeargs = {**pipeargs, "seed_tiles": generation_arguments["seed_tiles"]}
    if "seed_tiles_mode" in generation_arguments: pipeargs = {**pipeargs, "seed_tiles_mode": generation_arguments["seed_tiles_mode"]}
    if "seed_reroll_regions" in generation_arguments: pipeargs = {**pipeargs, "seed_reroll_regions": generation_arguments["seed_reroll_regions"]}
    image = pipe(**pipeargs)["sample"][0]
    image.save(generation_arguments["output"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a stable diffusion grid using a JSON file with all configuration parameters.')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('output', type=str, help='Output file')
    args = parser.parse_args()
    with open(args.config, "r") as f:
        generation_arguments = json.load(f)
    generation_arguments["output"] = args.output
    generate_grid(generation_arguments)
