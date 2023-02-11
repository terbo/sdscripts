# recurse through the output folder of Automatic 1111
# insert each image into a tinydb database
# present a UI for searching through images similar to Image Browser

import os, re, sys
import json, argparse
import gradio as gr
from tinydb import TinyDB, Query
from PIL import Image

negative_re = re.compile(r'Negative prompt:\s*(.*).*Steps')
step_re = re.compile(r'Steps:\s*(\d+)')
sampler_re = re.compile(r'Sampler:\s*([^,]*),?')
cfgscale_re = re.compile(r'CFG scale:\s*([\d.]*),?')
seed_re = re.compile(r'Seed:\s*(\d*),?')
modelhash_re = re.compile(r'Model hash:\s*([0-9A-Za-z]*)')
denoise_re = re.compile(r'Denoising strength: ([0-9.]*),?')

class AutomaticImage:
  def __init__(self, filename):
    self.fname = os.path.basename(filename)
    
    self.img = Image.open(filename)
    self.path = os.path.dirname(filename) or os.path.abspath(os.path.curdir)
    self.ctime = os.path.getctime(filename)
    self.prompt = ''
    self.negative = ''
    
    self.modelhash = ''
    self.model = ''
    
    self.seed = 0
    self.steps = 0
    self.cfgscale = 0
    self.denoise = 0
    self.sampler = ''

    self.img.load()
    self.resolution = (self.img.width, self.img.height)
    self.size = os.path.getsize(filename)

    self.params = self.parse_params()
  
  def i(self):
    return {
      'filename': f'{self.fname}',
      'path': f'{self.path}',
      'ctime': f'{self.ctime}',
      'size': f'{self.size}',
      'resolution': f'{self.resolution}',
      'prompt': f'{self.prompt}',
      'negative': f'{self.negative}',
      'sampler': f'{self.sampler}',
      'model': '',
      'modelhash': f'{self.modelhash}',
      'seed': f'{self.seed}',
      'steps': f'{self.steps}',
      'cfgscale': f'{self.cfgscale}'
    }
  
  def __str__(self):
    print(f"filename: {self.fname} in {self.path}\n")
    print(f"ctime: {self.ctime}")
    print(f"size: {self.size}")
    print(f"resolution: {self.resolution}")
    print(f"prompt: {self.prompt}")
    print(f"negative: {self.negative}")
    print(f"sampler: {self.sampler}")
    print(f"model: {self.model}")
    print(f"modelhash: {self.modelhash}")
    print(f"seed: {self.seed}")
    print(f"steps: {self.steps}")
    print(f"cfgscale: {self.cfgscale}")
    '''print(f"denoise: {self.denoise}\n")'''
    return ''

  def parse_params(self):
    try:
      params = self.img.info['parameters'].split('\n')
    except:
      return {}

    self.prompt = params[0]
    pnginfo = ' '.join(params[1:])

    try: self.steps = int(step_re.search(pnginfo)[1])
    except Exception as e: pass

    try: self.negative = negative_re.search(pnginfo)[1]
    except Exception as e: pass
    
    try: self.modelhash = modelhash_re.search(pnginfo)[1]
    except Exception as e: pass
    #print(self.model)

    try: self.cfgscale = float(cfgscale_re.search(pnginfo)[1])
    except Exception as e: pass
    
    try: self.seed = int(step_re.search(pnginfo)[1])
    except Exception as e: pass

    try: self.denoise = float(denoise_re.search(pnginfo)[1])
    except Exception as e: pass
    
    try: self.sampler = sampler_re.search(pnginfo)[1]
    except Exception as e: pass
    
class SDModel:
  def __init__(self, model):
    pass

def recurse_outputs(root_dir, func, imgdb_atime, model_cache):
  total_images = 0
  
  for root, dirs, files in os.walk(root_dir):
    for file in files:
      file_path = os.path.join(root, file)
      try:
        if os.path.splitext(file_path)[-1] in ['.jpg','.png']:
          # check lastmod of imgdb and file and skip based
          if imgdb_atime < os.path.getctime(file_path):
            img = AutomaticImage(file_path).i()
            hash = img['modelhash']
            img['model'] = model_cache[hash]
            func(img)

            total_images += 1
      except Exception as e:
        pass
        #print(f'Reading {file_path}: {e}')
  return total_images

def load_model_cache(model_cache_path):
  result = {}
  
  try:
    model_cache = json.loads(''.join(open(model_cache_path).readlines()))
    
    for model in model_cache['hashes'].keys():
      hash = model_cache['hashes'][model]['sha256'][0:10]
      print(f'{model}: {hash}')
      result[hash] = os.path.basename(model)
  except:
      print('No model cache.')
      return False

  return result

def image_search(q):
  print(f'Search: {q}')
  
  Q = Query()
  results = [os.path.join(img['path'], img['filename'])
            for img in db.search(Q.prompt.matches(q))]

  print(str(len(results)) + ' results.')

  return results

def run(install_path):
  imgdb_path =  os.path.join(install_path, 'imgdb.json')
  outputs_path = os.path.join(install_path,'outputs')
  model_cache_path = os.path.join(install_path, 'cache.json')
 
  model_cache = load_model_cache(model_cache_path) 
  
  try:
    imgdb_atime =  os.path.getatime(imgdb_path)
  except:
    imgdb_atime = 0

  db = TinyDB(imgdb_path)
  total_images = recurse_outputs(outputs_path, db.insert, imgdb_atime, model_cache)
  
  new = ''
  if imgdb_atime != 0: new = ' new'
  output = f'Found {total_images}{new} images.'

  print(output)

  # settings: image rows/columns, size
  # search

  return db

def ui(db):
  with gr.Blocks() as demo:
    with gr.Tab("Images"):
      with gr.Row():
        search = gr.Textbox(
            label="Enter your search",
            show_label=False,
            max_lines=1,
            placeholder="Enter your search",
        ).style(container=False)
        search_select = gr.CheckboxGroup(["Prompts", "Checkpoints"], label="Search type"), 
        btn = gr.Button("Search").style(full_width=False)
    #  search = gr.Interface(fn=image_search, inputs="text", outputs="text")
      
      with gr.Row():
        gallery = gr.Gallery(show_label=True).style(grid=[8], height='auto')

  
    with gr.Tab("Prompts"):
      squared = gr.Number(value=1)
    
    with gr.Tab("Settings"):
      crossed = gr.Number(value=2)

    btn.click(image_search, search, gallery)
  demo.launch()

if __name__ == '__main__':
  # input: automatic install directory
  # recurse directories
  # options: image directory, model cache path, imgdb path

  if len(sys.argv) < 2:
    print(f'Organize Automatic 1111 Web UI Images into a TinyDB database.')
    print(f'Usage: {sys.argv[0]} [webui-path]')
    sys.exit()
  
  db = run(sys.argv[1])
  ui(db)
