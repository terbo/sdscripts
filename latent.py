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

# TODO: Add CSV
# TODO: Remember last seed to make merge more coherent
	(Or remove the last frames that are slightly out of sync with last generation)
# TODO: Add extension parsing, so .mp4 isn't the default

import os, sys, random, argparse

# Spawn latent blending
def main(opts):
  lb = LatentBlending(sdh)
  lb.load_branching_profile(quality=opts.quality, depth_strength=opts.depth)

  lb.set_width(opts.width)
  lb.set_height(opts.height)

  prompts = opts.prompt
  prompts.reverse()
  last_prompt1 = None

  total_videos = len(opts.prompt)
  current_video = 0
  videos = []
  
  last_seed1 = None
  last_seed2 = None

  # iterate through prompts if there are more than 2 prompts,
  # and take the last prompt and generated seed and make them prompt1 and seed1
  while len(prompts) > 0:
    current_video += 1
    
    if not last_seed1:
      last_seed1 = random.randint(1, int(float("1e12")))
    else:
      last_seed1 = last_seed2
    
    if not last_prompt1:
      prompt1 = prompts.pop()
      last_prompt1 = prompt1
    else:
      last_prompt1 = last_prompt2
    
    if opts.style:
      last_prompt1 += ', ' + opts.style
    
    try:
      prompt2 = prompts.pop()
      last_prompt2 = prompt2
      last_seed2 = random.randint(1, int(float("1e12")))
      
    
    except:
      last_prompt2 = prompt1
      last_seed2 = last_seed1
    
    if opts.style:
      last_prompt2 += ', ' + opts.style
    
    lb.set_prompt1(last_prompt1)
    lb.set_prompt2(last_prompt2)

    fixed_seeds = [last_seed1, last_seed2]
    
    if opts.debug:
      print('\nBlending "%s" (%d) and "%s" (%d) (video %d)\n' % (last_prompt1, last_seed1, last_prompt2, last_seed2, current_video))
    
    # Run latent blending
    imgs_transition = lb.run_transition(fixed_seeds=fixed_seeds)

    # Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
    imgs_transition_ext = add_frames_linear_interp(imgs_transition, int(opts.duration), int(opts.fps))
    
    if opts.video:
      output = '%s_%d.mp4' % (opts.video, current_video)

      if opts.debug:
        print('\nMaking movie from %s' % output)
        make_movie(opts, imgs_transition_ext, output)
      
      if total_videos > 2:
        videos.append(output)
      
  if opts.video and total_videos > 1:
    print('Merging %d videos into %s %s' % (len(videos), opts.video, list(videos)))
    concatenate_movies(opts.video + '.mp4', videos)
    
  else:
    print('opts.video: %s / videos: %d' % (opts.video, len(videos)))
    print('Saving %d images' % len(imgs_transition_ext))

    imgnum = 0
    for img in imgs_transition_ext:
      imgnum += 1
      num = format(imgnum, '03d')
      #img.save('%s_%s.png' % (opts.name, num))

# Save as MP4
def make_movie(opts, imgs, output):
    if os.path.isfile(output):
        if opts.debug:
          print('\nRemoving %s\n' % output)
        os.remove(output)
    ms = MovieSaver(output, fps=opts.fps, shape_hw=[sdh.height, sdh.width])
    for img in tqdm(imgs):
        ms.write_frame(img)
    ms.finalize()

def parse_args():
  parser = argparse.ArgumentParser(prog='latent',
                                   description='Blend prompts with latent diffusion',
                                   epilog='https://github.com/lunarring/latentblending/, http://github.com/terbo/latent.py')
  
  parser.add_argument('-m', '--model', help='model to load', default='sd-v2-1-768-ema-pruned.ckpt')
  parser.add_argument('-v1', action='store_true', help='use v1 model (v1-inference.yaml must be in model directory')
  parser.add_argument('-n', '--name', help='image basename to save if no video name provided')
  parser.add_argument('-q', '--quality', help='preset quality (lowest, low, medium, high, ultra)', default='medium')
  parser.add_argument('-d', '--depth', help='depth of diffusion iterations', default=0.65)
  parser.add_argument('-p','--prompt', action='append', help='text prompt to generate. at least 2 required, no limit')
  parser.add_argument('-w','--width', help='image/video width')
  parser.add_argument('-H','--height', help='image/video height')
  #parser.add_argument('--seed', action='append', help='seed for each prompt, if you leave any out they will be random')
  parser.add_argument('--style', help='style (appended to the end of each prompt)')
  parser.add_argument('-v','--video', help='generate video, with filename (leave extension off)', default='latent.mp4')
  parser.add_argument('-f','--fps', help='frames per second of video', default=5)
  parser.add_argument('-l', '--duration', help='duration of video', default=5)
  parser.add_argument('--debug', action='store_true', help='print debug information')
  
  return (parser.parse_args(), parser.print_help)

if __name__ == '__main__':
  (opts, usage) = parse_args()

  if not opts.prompt:
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
    sdh = StableDiffusionHolder(opts.model, os.path.dirname(opts.model) + '/v1-inference.yaml')
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
  print(opts)

  main(opts)
