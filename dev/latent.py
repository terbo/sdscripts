# Copyright 2022 Lunar Ring. All rights reserved.
# Written by Johannes Stelzer, email stelzer@lunar-ring.ai twitter @j_stelzer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [terbo 1/20/2023]
# Latent blending with infinite prompts v.002

# Modified from https://github.com/lunarring/latentblending
# Requires: latentblending, stable diffusion checkpoint (v1 or v2), 8GB VRAM

# Usage: latent.py -p "first image prompt" -p "second image prompt" [-p "more prompts"]
# Run latent.py -h for more information

'''
  Example usage:

# using the openjourney model, generate a video with at 15 FPS with 10 second transitions and a fixed seed
latent.py -v1 -m c:\models\mdjrny-v4.ckpt --style "evolving on planet earth, in the style of mdjrny-grfft" \
    --fps 15 --length 7 \
    -p "the motivated microbe" -p "the copius cells" -p "the mighty mitochondria" -p "the crafty frog"
    -p "the curious monkey"-p "the bumbling man" -p "the perfected angel" -p "the omniscient watcher" \
    --seed 970000079 --fixed --name evolution_of_an_illusion

# using the midjourney graffiti model, generate a video at 30 FPS with 7 second transitions and iterating seeds
latent.py -v1 -m d:\models\mdjrny-grfft.ckpt  --fps 30 --length 7 \
    -p "blank expression on his face" -p "a bored look on his face" -p "a slight smile on his face" \
    -p "a slightly happy look on his face" -p "a happy look on his face" -p "a wide smile on his face" \
    -p "a very happy look on his face" -p "an ecstatic look on his face" -p "a closeup with a grin" \
    --seed 14500541 --fixed --name graffiti_with_shadows \
    --style "a full body shot of the shadowy dark man, wearing black cargo pants, a black leather jacket, and a white t-shirt, with a large redish-brown afro, a large gold chain around his neck, arms crossed, in the style of mdjrny-grfft"

# using the dreamlike diffusion model, generate a 640x480 video at 60 FPS with 15 second transitions and random seeds,
#   with negative prompts for text, saving generation details to a text file
latent.py -v1 -m dreamlike-diffusion-v1.0.mp4 --fps 60 --length 15 \
    -p "new moon" -p "waxing crescent moon" -p "first quarter moon" -p "waxing gibbous moon" -p "full moon" \
    -p "waning gibbous moon" -p "last quarter moon" -p "waning crescent moon" \
    --negative "words, characters, text, alphabet" \
    --style "establishing shot of a still, calm ocean horizon, dreamlikeart" -t --name lunar_calender

# note, above examples may not be all that great right now
'''

# TODO: Add txt/json/csv input for parameters, prompts
# TODO: Add individual image saving
#       (convert np array to image, modifying latent_blending.py, or ffmpeg img extraction)
# TODO: Implement upscaling, inpainting (hows that work??)
# TODO: Rename script blend.py

# TODO: Fix examples
# TODO: Fix seed stepping, add 
# TODO: Interface for branch1_influence, guidance_scale, guidance_scale_mid_damper, num_inference_steps, nmb_trans_images

# DONE: Fixed seed, stepping
# DONE: Add txt/csv generation info saving
# DONE: Save .txt file with prompts, seeds, other generation information
# DONE: Add extension parsing, so .mp4 isn't the default
# DONE: Remove the last frames that are slightly out of sync with next generation
# DONE: Just use run_multi_transition (derp) and give an option to extract frames from the video.

import os, sys, random, argparse
from datetime import datetime as dt

# Spawn latent blending
def main(opts):
  lb = LatentBlending(sdh)
  lb.load_branching_profile(quality=opts.quality, depth_strength=opts.depth)

  lb.set_width(opts.width)
  lb.set_height(opts.height)
 
  if len(opts.negative):
    lb.set_negative_prompt(opts.negative)

  session = os.path.splitext(opts.name)[0]
  
  prompts = []
  seeds = []

  if opts.style:
    for prompt in opts.prompt:
      prompts.append('%s, %s' % (prompt, opts.style))
  
  else:
    prompts = opts.prompts

  if not opts.seed:
    for s in range(0, len(prompts)):
      seeds.append(random.randint(1, int(float("1e12"))))
 
  elif opts.fixed_seed:
    while len(seeds) < (len(prompts)):
      seeds.append(opts.seed[0])
    
    if opts.debug:
      print('%s == %s' % (len(prompts), len(seeds)))
    
  elif opts.step_seed and len(opts.seed):
    seeds = opts.seed

    while len(seeds) < len(opts.prompt):
      seeds.append(seeds[-1] + opts.step_val)

  elif len(seeds) < len(opts.prompt):
    while len(seeds) < len(opts.prompt):
      seeds.append(random.randint(1, int(float("1e12"))))

  elif len(opts.seed) > len(opts.prompt):   # why? but why not..
    seeds = opts.seed[0:len(opts.prompt)]

  # time the generation
  start = dt.now()

  lb.run_multi_transition(opts.name, prompts, seeds,
                          fps=opts.fps, duration_single_trans=opts.length)

  end = dt.now()
  
  if opts.savetext:
    save_text(session, opts, prompts, seeds)
  
  if opts.savejson:
    import json
    save_json(session, opts, prompts, seeds)

  # may be better to do this in by modifying the sub
'''
  if opts.images:
    imgnum = 0
    
    for img in imgs_transition_ext:
      imgnum += 1
      num = format(imgnum, '03d')
      #img.save('%s_%s.png' % (opts.name, num))
'''

def save_text(session, opts, prompts, seeds):
  with open('%s.txt' % session, 'w') as fp_text:
    fp_text.write("Session '%s': %s [%s/%s] @ %s FPS, %ss transition duration, %d depth, %s quality\n" %
                  (session, opts.model, opts.width, opts.height, opts.fps, opts.length, opts.depth, opts.quality))

    fp_text.write("Style: %s\n" % opts.style)
    fp_text.write("Negative prompt: %s\n" % opts.negative)

    opts.seed.reverse()

    for prompt in opts.prompt:
      fp_text.write("%s\n%s\n\n" % (prompt, seeds.pop()))

def save_json(session, opts, prompts, seeds):
  cur_prompt = 1
  prompt_list = []

  for prompt in prompts:
    prompt_list[cur_prompt] = {prompt: seeds.pop(0)}
    cur_prompt += 1

  params = { 'session': session, 'model': opts.model, 'quality': opts.quality,
             'fps': opts.fps, 'width': opts.width, 'height': opts.height,
             'transition': opts.length, 'style': opts.style,
             'negative': opts.negative, 'prompts': prompt_list, 'depth': opts.depth
           }

  with open('%s.json' % session, "w") as fp_json:
    json.dump(params, fp_json)
  
def parse_args():
  parser = argparse.ArgumentParser(prog='latent',
                                   description='Blend prompts with latent diffusion',
                                   epilog='based on https://github.com/lunarring/latentblending/, by http://github.com/terbo/sdscripts/')
  
  parser.add_argument('-m', '--model', help='model to load', default='sd-v2-1-768-ema-pruned.ckpt')
  parser.add_argument('-v1', action='store_true', help='use v1 model (v1-inference.yaml must be in model directory')
  
  parser.add_argument('-n', '--name', help='session/video name, if generating images, a directory will be created')
  parser.add_argument('-q', '--quality', help='preset quality (lowest, low, medium, high, ultra)', default='medium')
  parser.add_argument('-d', '--depth', type=float, help='depth of diffusion iterations', default=0.65)
  parser.add_argument('-p', '--prompt', action='append', help='text prompt to generate. at least 2 required')
  parser.add_argument('-w', '--width', type=int, help='image/video width, must be multiple of 64')
  parser.add_argument('-H', '--height', type=int, help='image/video height, must be multiple of 64')
  
  parser.add_argument('--seed', action='append', type=int, help='seed for each prompt, must be same number as prompts')
  parser.add_argument('--step', action='store_true', help='increment the seed of each image by 1')
  parser.add_argument('--step-val', type=int, default=1, action='store_true', help='set seed step increment')
  parser.add_argument('--fixed', action='store_true', help='use a fixed seed for each image')
  
  parser.add_argument('--style', help='style (appended to the end of each prompt)', default='')
  parser.add_argument('--negative', help='negative prompt', default='')
  
  parser.add_argument('-f','--fps', type=int, help='frames per second of video', default=5)
  parser.add_argument('-l', '--length', type=int, help='duration of transition', default=5)
  #parser.add_argument('-i', '--images', help='extract images from video and place in session directory')
  
  parser.add_argument('-t', '--savetext', action='store_true', help='save generation details in text format')
  parser.add_argument('-j', '--savejson', action='store_true', help='save generation details in json format')
  
  parser.add_argument('--debug', action='store_true', help='print debug information')
  
  return (parser.parse_args(), parser.print_help)

if __name__ == '__main__':
  (opts, usage) = parse_args()

  if (not opts.prompt) or (not opts.name):
    usage()
    sys.exit()

  import torch
  torch.backends.cudnn.benchmark = False
  import numpy as np
  import warnings
  warnings.filterwarnings('ignore')
  import warnings
  import torch
  from tqdm.auto import tqdm
  from PIL import Image
  # import matplotlib.pyplot as plt
  import torch
  from movie_util import MovieSaver, concatenate_movies
  from typing import Callable, List, Optional, Union
  from latent_blending import LatentBlending, add_frames_linear_interp
  from stable_diffusion_holder import StableDiffusionHolder

  torch.set_grad_enabled(False)
  
  if(opts.v1):
    sdh = StableDiffusionHolder(opts.model, '%s/v1-inference.yaml' % os.path.dirname(opts.model))
    
    if not opts.width:
      opts.width = 512
    if not opts.height:
      opts.height = 512
  else:
    sdh = StableDiffusionHolder(opts.model)
    
    if not opts.width:
      opts.width = 768
    if not opts.height:
      opts.height = 768

  quality = opts.quality
  
  if opts.debug:
    print(opts)

  main(opts)
