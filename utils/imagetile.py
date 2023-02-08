from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
from io import StringIO
from PIL import Image
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import images, paths, sd_samplers, processing, sd_models, sd_vae
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
from PIL import Image
import glob
import os
import re
import sys

screen_dims = [(3840,2160), (1920,1080), (1366,768), (360,640), (414,896), (1536,864), (375,667)]

def imagetile(p, img_size, tosize):
  # Opens an image
  bg = p.images[0]

  # The width and height of the background tile
  bg_w, bg_h = bg.size
  print(f"Got image {bg_w}x{bg_h}")
  # Creates a new empty image, RGB mode, and size 1000 by 1000
  new_im = Image.new('RGB', tosize)

  # The width and height of the new image
  w, h = new_im.size
  print(f"Created image {w}x{h}")

  # Iterate through a grid, to place the background tile
  r = [x * .01 for x in img_size]
  print(r)
  #r.reverse()
  d = 1
  for i in r:
    new_h = int(bg_h * i)
    new_w = int(bg_w * i)
    new_bg = bg.resize((new_h, new_w))
    print(i)
    print('%d/%d ' % (new_h, new_w)),

    for i in range(0, w, new_w):
      for j in range(0, h, new_h):
        # Change brightness of the images, just to emphasise they are unique copies
        #bg = Image.eval(new_bg, lambda x: x+(i+j)/1000)

        #paste the image at location i, j:
        new_im.paste(new_bg, (i, j))
    p.images.append(new_im)
    d += 1
  return p

class Script(scripts.Script):
  def title(self):
    return "Image Tile"
  
  def show(self, is_img2img):
    return True # is_img2img

  def ui(self, is_img2img):
    with gr.Column(scale=19):
      with gr.Row():
          with gr.Row(variant="compact", elem_id=self.elem_id("tile_size")):
            tiny_image = gr.Checkbox(label='Tiny', value=True, elem_id=self.elem_id("tiny_image"))
            mini_image = gr.Checkbox(label='Mini', value=True, elem_id=self.elem_id("mini_image"))
            small_image = gr.Checkbox(label='Small', value=True, elem_id=self.elem_id("small_image"))
            medium_image = gr.Checkbox(label='Medium', value=True, elem_id=self.elem_id("medium_image"))
            large_image = gr.Checkbox(label='Large', value=False, elem_id=self.elem_id("large_image"))
    
      with gr.Row():
          screen_dim = gr.Dropdown(label="Full Resolution", choices=[x for x in screen_dims], value=str("(1920,1080)"), type="value", elem_id=self.elem_id("screen_dim"))

    return [screen_dim, mini_image, tiny_image, small_image, medium_image, large_image]

  def run(self, p, screen_dim, mini_image, tiny_image, small_image, medium_image, large_image):
    #processing.fix_seed(p)

    #if tiny_image:     imgsize = (128,128)
    #elif mini_image:   imgsize = (256,256)
    #elif small_image:  imgsize = (384,384)
    #elif medium_image: imgsize = (512,512)
    #elif large_image:  imgsize = (768,768)
    img_size = [] 
    if tiny_image:   img_size.append(20)
    if mini_image:   img_size.append(40)
    if small_image:  img_size.append(60)
    if medium_image: img_size.append(80)
    if large_image:  img_size.append(90)
    
    print(f"[ImageTile]: Creating tiles {img_size} into {screen_dim} Image")

    grid_infotext = [None]

    processed = processing.process_images(p)
   
    processed = imagetile(processed, img_size, screen_dim)

    print("[ImageTile]: Done.")
    return processed
