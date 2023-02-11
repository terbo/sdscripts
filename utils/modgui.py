import os, glob, time
import gradio as gr
import numpy as np
from PIL import Image
import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler
from diffusiontools.tiling import StableDiffusionTilingPipeline

def mixture_of_diffusers(prompts, seed, gc, steps, amount, model_id, tile_width, tile_height, tile_row_overlap, tile_col_overlap, cpu_vae, sampler):
  prompts = [prompts.split('\n')]
  amount = int(amount)
  steps = int(steps)
  gc = int(gc)
  
  if seed == -1:
    seed = generate_seed()
  else:
    seed = int(seed)

  final_height = (len(prompts[0]) * tile_width) - (tile_row_overlap * (len(prompts[0]) - 1))
  imgsize = '%dx%d' % (final_height, tile_height)
  
  print(f'Prompt: {prompts}')
  print(f'Model: {model_id}')
  print(f'Steps/images: {steps}, {amount}')
  print(f'Seed/GC: {seed}, {gc}')
  print(f'Tile Width/Height: {tile_width}x{tile_height}')
  print(f'Overlap Width/Height: {tile_row_overlap}x{tile_col_overlap}')

  # Prepared scheduler
  if sampler == "DDIM":
    scheduler = DDIMScheduler()
  elif sampler == "LMS":
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)
  else:
    raise ValueError(f"Unrecognized scheduler {sampler}")
  pipe = StableDiffusionTilingPipeline.from_pretrained(model_id, scheduler=scheduler, use_auth_token=True).to('cuda:0')

  #if "seed_tiles" in generation_arguments: pipeargs = {**pipeargs, "seed_tiles": generation_arguments["seed_tiles"]}
  #if "seed_tiles_mode" in generation_arguments: pipeargs = {**pipeargs, "seed_tiles_mode": generation_arguments["seed_tiles_mode"]}
  #if "seed_reroll_regions" in generation_arguments: pipeargs = {**pipeargs, "seed_reroll_regions": generation_arguments["seed_reroll_regions"]}
 
  images = []
    
  pipeargs = {
      'guidance_scale': gc,
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
    image = pipe(**pipeargs)["sample"][0]
    end_time = int(time.time())
  
    output = f"outputs/{end_time}_{pipeargs['seed']}.png"
    print(f'Generated image to {output} in {end_time - start_time}s')
    images.append(image)
    image.save(output)
    
    params = '%s\n' % prompts[0]
    params += 'Steps: %s, Sampler: %s, CFG scale: %s, Seed: %s, ' % (steps, sampler, gc, pipeargs['seed'])
    params += 'Size: %s, Model: %s\n' % (imgsize, model_id)
    params += 'Tile row overlap: %d Tile column overlap: %d\n' % (tile_row_overlap, tile_col_overlap)
    params += 'CPU VAE: %s' % cpu_vae
  
    params_file = os.path.splitext(output)[0] + '.txt'
    
    with open(params_file, 'w') as params_fp:
      params_fp.write(params)

    pipeargs['seed'] += 100
    # add generation params to pnginfo..

  return images

def find_output_images():
  output_images = glob.glob('outputs/*png')
  output_images = reversed(output_images)
  return output_images

def reset_seed():
  return -1

def generate_seed():
  return int(np.random.randint(999999999))

def ui():
  find_output_images()
  
  with gr.Blocks(css='#gallery { max-width: 15% } ') as demo:
    with gr.Column():
      with gr.Row():
        with gr.Column():
          with gr.Row():
            prompts = gr.Textbox(label='Prompt(s)')
            with gr.Column():
              tile_width = gr.Slider(64, 768, value=384, step=64, label='Tile Width', interactive=True)
              tile_height = gr.Slider(64, 768, value=512, step=64, label='Tile Height', interactive=True)
              tile_row_overlap = gr.Slider(64, 768, value=256, step=64, label='Overlap Width', interactive=True)
              tile_col_overlap = gr.Slider(64, 768, value=256, step=64, label='Overlap Height', interactive=True)
          
          with gr.Row():
            model_id = gr.Dropdown(['CompVis/Stable-Diffusion-1-4', 'prompthero/openjourney-v2'], value='prompthero/openjourney-v2', label='Model')
            sampler = gr.Radio(['DDIM','LMS'], value='LMS', label='Sampler')
            
            seed = gr.Number(label='Seed', value=-1)
            new_seed = gr.Button(value='New')
            zero_seed = gr.Button(value='Old')
            
            gc = gr.Number(label='GC', value=7)
            steps = gr.Number(value=50, label='Steps')
            amount = gr.Number(value=1, label='Images to generate')
            cpu_vae = gr.Checkbox(value=False, label='Use CPU VAE')
          
          btn = gr.Button(value='Generate')
        outputs = gr.Gallery(elem_id='gallery', value=find_output_images())
    
    with gr.Column():
      output_image = gr.Gallery(label='Output').style(container=False) 

    new_seed.click(generate_seed, None, [seed])
    zero_seed.click(reset_seed, None, [seed])
    
    btn.click(mixture_of_diffusers, [prompts, seed, gc, steps, amount, model_id,
                                     tile_width, tile_height, tile_row_overlap, tile_col_overlap, cpu_vae, sampler], [output_image])
    
    #output_image.change(find_output_images, None, outputs)
    
    demo.launch()

if __name__ == '__main__':
  if not os.path.exists('outputs'):
    os.mkdir('outputs')
  
  ui()
