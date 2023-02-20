# mixture of diffusers gradio interface
# UI for https://github.com/albarji/mixture-of-diffusers
# v0.1.1 terbo - https://github.com/terbo/sdscripts/utils
#
# Mixture of Diffusers:
#  ... a method for integrating a mixture of different diffusion processes collaborating to generate a single image.
#  Each diffuser focuses on a particular region on the image, taking into account boundary effects to promote a smooth blending.
#
#
# Install MoD, save this script to the mixtures-of-diffusion directory,
# and either use the automatic1111 venv (will just need to install diffusers package),
# or:
#
#   create a venv (python -mvenv venv --prompt mod)
#   enter the venv (venv\scripts\activate)
#   install packages (pip install -r requirements.txt)
#   install torch, using the same method you use for automatic 1111
#
# Saves images in outputs/, and besides each saves a text file containing generation parameters.
#
# Some models don't work properly, I think it may be a VAE issue, but the ones in the dropdown seem OK.
# I usually img2img in automatic1111 after generating here, as the colors usually aren't too crisp.
# 
# Updated to reflect latest MoD commits, still messing with gradio.
#
# TODO
#
# fix gradio layout...
# fix output gallery
# add img2img support
# add generation params to pnginfo..
# make images into grid preview when rendered
# switch to output window when when generate is clicked

import os, glob, time
import gradio as gr
import numpy as np
from PIL import Image
import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler, KarrasVeScheduler, EulerDiscreteScheduler, DDPMScheduler
from mixdiff.tiling import StableDiffusionTilingPipeline

OUTPUT = './outputs'

schedulers = ['LMSDiscreteScheduler', 'DDIMScheduler'] #, 'KarrasVeScheduler', 'EulerDiscreteScheduler', 'DDPMScheduler']

models = [
 'None (input text above)',
 'CompVis/Stable-Diffusion-1-4',
 'runwayml/stable-diffusion-v1-5',
 'prompthero/openjourney-v2',
 'Envvi/Inkpunk-Diffusion',
 'XpucT/Deliberate',
 'nitrosocke/archer-diffusion',
 'ShadoWxShinigamI/midjourney-graffiti'
]

def mixture_of_diffusers_img2img(prompts, img2img_image, styles, seed, seedstep, cfgscale, steps,
                                 amount, model_id, model_id_in, tile_width, tile_height,
                                 tile_row_overlap, tile_col_overlap, cpu_vae, sampler):
  pass

def mixture_of_diffusers(prompts, styles, seed, seedstep, cfgscale, steps, amount,
                         model_id, model_id_in, tile_width, tile_height,
                         tile_row_overlap, tile_col_overlap, cpu_vae, sampler):
  amount = int(amount)
  seedstep = int(seedstep)
  steps = int(steps)
  cfgscale = float(cfgscale)

  if len(styles):
    prompts = [list(map(lambda x: x + f', {styles}', prompts.split('\n'))) if len(styles) else '']
  else:
    prompts = [prompts.split('\n')]

  if seed == -1 or seed == '':
    seed = generate_seed()
  else:
    seed = int(seed)

  final_width = (len(prompts[0]) * tile_width) - (tile_row_overlap * (len(prompts[0]) - 1))
  imgsize = '%dx%d' % (final_width, tile_height)

  if model_id.startswith('None'):
    if len(model_id_in):
      model_id = model_id_in
    else:
      model_id = models[0]

  print(f'Prompt: {prompts}')
  print(f'Model/Sampler: {model_id}/{sampler}')
  print(f'Steps/images: {steps}, {amount}')
  print(f'Seed/CFG: {seed}, {cfgscale}')
  print(f'Final Resolution: {final_width}x{tile_height}')
  print(f'Tile Width/Height: {tile_width}x{tile_height}')
  print(f'Overlap Width/Height: {tile_row_overlap}x{tile_col_overlap}')

  # Prepared scheduler
  if sampler == 'DDIMScheduler':
    scheduler = DDIMScheduler()
  elif sampler == 'DDPMScheduler':
    scheduler = DDPMScheduler()
  elif sampler == 'KarrasVeScheduler':
    scheduler = KarrasVeScheduler()
  elif sampler == 'EulerDiscreteScheduler':
    scheduler = EulerDiscreteScheduler()
  elif sampler == 'LMSDiscreteScheduler':
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)
  else:
    raise ValueError(f'Unrecognized scheduler {sampler}')
  pipe = StableDiffusionTilingPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to('cuda:0')

  #if 'seed_tiles' in generation_arguments: pipeargs = {**pipeargs, 'seed_tiles': generation_arguments['seed_tiles']}
  #if 'seed_tiles_mode' in generation_arguments: pipeargs = {**pipeargs, 'seed_tiles_mode': generation_arguments['seed_tiles_mode']}
  #if 'seed_reroll_regions' in generation_arguments: pipeargs = {**pipeargs, 'seed_reroll_regions': generation_arguments['seed_reroll_regions']}

  images = []

  pipeargs = {
      'guidance_scale': cfgscale,
      'num_inference_steps': steps,
      'seed': seed,
      'prompt': prompts,
      'tile_width': tile_width,
      'tile_height': tile_height,
      'tile_row_overlap': tile_row_overlap,
      'tile_col_overlap': tile_col_overlap,
      'guidance_scale_tiles': None,
      'cpu_vae': cpu_vae
  }

  for i in range(amount):
    start_time = int(time.time())
    image = pipe(**pipeargs)['sample'][0]
    end_time = int(time.time())

    output = f"{OUTPUT}/{end_time}_{pipeargs['seed']}.png"
    print(f'Generated image to {output} in {end_time - start_time}s')
    images.append(image)
    image.save(output)

    params = '%s\n' % prompts[0]
    params += 'Steps: %s, Sampler: %s, CFG scale: %s, Seed: %s, ' % (steps, sampler, cfgscale, pipeargs['seed'])
    params += 'Size: %s, Model: %s\n' % (imgsize, model_id)
    params += 'Tile row overlap: %d Tile column overlap: %d\n' % (tile_row_overlap, tile_col_overlap)
    params += 'CPU VAE: %s' % cpu_vae

    params_file = os.path.splitext(output)[0] + '.txt'

    with open(params_file, 'w') as params_fp:
      params_fp.write(params)

    pipeargs['seed'] += seedstep

  return images

def find_output_images():
  output_images = glob.glob(f'{OUTPUT}/*png')
  output_images = reversed(output_images)
  return output_images

def reset_seed():
  return -1

def generate_seed():
  return int(np.random.randint(999999999))

def ui():
  output_images = find_output_images()

  with gr.Blocks(analytics_enabled=False, title='Mixture of Diffusers') as demo:
    with gr.Row():
      with gr.Column(scale=2):
        prompts = gr.Textbox(label='Prompt(s)', lines=2)
        style = gr.Textbox(label='Style (added to every prompt)')
        model_id_in = gr.Text(value='', label='Model')
        model_id = gr.Dropdown(models, value='runwayml/stable-diffusion-v1-5', label='Model')
        cpu_vae = gr.Checkbox(value=True, label='Use CPU VAE')
        tile_width = gr.Slider(32, 1024, value=512, step=32, label='Tile Width', interactive=True)
        tile_height = gr.Slider(32, 1024, value=512, step=32, label='Tile Height', interactive=True)
        tile_row_overlap = gr.Slider(32, 1024, value=256, step=32, label='Overlap Width', interactive=True)
        tile_col_overlap = gr.Slider(32, 1024, value=256, step=32, label='Overlap Height', interactive=True)
        amount = gr.Number(value=1, label='Images to generate')
        steps = gr.Number(value=50, label='Steps')
        sampler = gr.Radio(schedulers, value=schedulers[0])
        cfgscale = gr.Slider(1,30, step=0.5, value=7, label='CFG Scale')
        seedstep = gr.Number(label='Seed step', value=50)
        seed = gr.Number(label='Seed', value=-1)
        new_seed = gr.Button(elem_id='new_seed', value='New Seed')
        random_seed = gr.Button(elem_id='random_seed', value='Random Seed')

    generate_btn = gr.Button(value='Generate')

    with gr.Row():
      with gr.Tab('Output'):
        output_image = gr.Gallery(label='Output')

      with gr.Tab('img2img'):
        img2img_btn = gr.Button(value='Generate')
        img2img_image = gr.Image(label='Input')

      with gr.Tab('Results'):
        update_btn = gr.Button(value='Refresh')
        outputs = gr.Gallery(value=output_images)

    new_seed.click(generate_seed, None, [seed])
    random_seed.click(reset_seed, None, [seed])

    update_btn.click(find_output_images, None, [outputs])

    generate_btn.click(mixture_of_diffusers, [prompts, style, seed, seedstep, cfgscale, steps, amount, model_id, model_id_in,
                                     tile_width, tile_height, tile_row_overlap, tile_col_overlap, cpu_vae, sampler], [output_image])

    img2img_btn.click(mixture_of_diffusers_img2img, [prompts, img2img_image, style, seed, seedstep, cfgscale, steps, amount, model_id, model_id_in,
                                     tile_width, tile_height, tile_row_overlap, tile_col_overlap, cpu_vae, sampler], [output_image])

    #output_image.change(find_output_images, None, outputs)

    demo.launch(show_api=False, show_error=True)

if __name__ == '__main__':
  os.makedirs(OUTPUT, exist_ok=True)

  ui()
