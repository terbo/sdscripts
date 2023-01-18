# Using Automatic 1111 WebUI, webuiapi and PIL,
# generate an image from the command line

# TODO: csv file reading, config file
#       img2img, loopback, scripts, hires fix

# BUGS: If the server crashes (or in my case the ssh connection times out)
#       the API blocks.

import argparse, string, random
import re, sys
import webuiapi
from PIL import Image

seed_base = 1e15 # 1000000000000000

def parse_args():
    parser=argparse.ArgumentParser(prog='generate',
                                   description='Partial command line interface for Automatic 1111 Web UI',
                                   epilog='http://github.com/terbo/sdscripts')
    parser.add_argument('--prompt', help='prompt to generate image from', default='happy little trees, style of bob ross')
    parser.add_argument('--prompts', help='read prompts from a text file')
#   parser.add_argument('--promptcsv', help='read prompts and generation options from a csv file, format incoming')

    parser.add_argument('--negative', help='negative prompt', default='')
    parser.add_argument('--steps', help='sampling steps', default=20)
    parser.add_argument('--seed', help='default seed', default=0)
    parser.add_argument('--cfg', help='CFG scale (0-30)', default=7)
    parser.add_argument('--sampler', help='sampler to use', default='DPM++ 2S a Karras')
    parser.add_argument('--style', help='saved style to apply, use --liststyles to view styles', default='')

    parser.add_argument('--batch', help='number of images to create', default=1)
    
    parser.add_argument('--host', help='WebUI API Host', default='omni')
    parser.add_argument('--port', help='WebUI API Port', default='7860')
    
    parser.add_argument('--width', help='Image width', default=512)
    parser.add_argument('--height', help='Image height', default=512)
    
    parser.add_argument('--getmodel', action='store_true', help='show default model in UI')
    parser.add_argument('--setmodel', help='set default model in UI')
    parser.add_argument('--listmodelsfull', action='store_true', help='list available models with all information')
    parser.add_argument('--listmodels', action='store_true', help='list available models (2)')
    
    parser.add_argument('--listsamplers', action='store_true', help='list available samplers')
    parser.add_argument('--liststyles', action='store_true', help='list available prompt styles')
   
    parser.add_argument('--getoptions', action='store_true', help='return current options')
    parser.add_argument('--showimg', action='store_true', help='show generated image')
    parser.add_argument('--showinfo', action='store_true', help='show generation information')
    
    opts=parser.parse_args()
    return opts

# from automatic1111 modules\images.py
def sanitize_filename_part(text, replace_spaces=True):
  invalid_filename_chars = '<>:"/\\|?*\n'
  invalid_filename_prefix = ' '
  invalid_filename_postfix = ' .'
  re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
  re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
  re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
  max_filename_part_length = 128

  if text is None:
      return None

  if replace_spaces:
      text = text.replace(' ', '_')

  text = text.translate({ord(x): '_' for x in invalid_filename_chars})
  text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
  text = text.rstrip(invalid_filename_postfix)
  return text

def generate_image(opts, negative_prompt='(bad_prompt_version2:0.8)'):
  if not opts.seed:
    seed = int(random.random() * 1e10)

  config = {
          'prompt': opts.prompt,
          'width': opts.width,
          'height': opts.height,
          'negative_prompt': negative_prompt,
          'cfg': opts.cfg,
          'sampler': opts.sampler,
          'steps': opts.steps,
          'seed': seed,
  }
  
  res = api.txt2img(prompt=config['prompt'],
                    negative_prompt=config['negative_prompt'],
                    width=config['width'],
                    height=config['height'],
                    seed=config['seed'],
                    styles=[],
                    cfg_scale=7,
                    sampler_index=config['sampler'],
                    steps=config['steps'],
                   )
  
  return (config, res)

def save_text_properties(config, opts, output_file):
  txtfile = open(output_file, 'w')
  text = [config['prompt'], "\n", "Steps: %d, Sampler: %s, CFG scale: %d, Seed: %d" %
        (config['steps'], config['sampler'], config['cfg'], config['seed'])]
  txtfile.writelines(text)
  txtfile.close()
  if opts.showinfo:
    print(''.join(text))


def generate(opts):
  
  def _generate(opts):
    prompt = opts.prompt

    config, res = generate_image(opts)

    output_file = ('%d_%s' % (config['seed'], prompt))
    image_name = sanitize_filename_part(output_file)
    res.image.save(image_name + '.png')
    save_text_properties(config, opts, image_name + '.txt')

    # if making a batch, create a collage of all images
    if opts.showimg:
      image=Image.open(image_name + '.png')
      image.show()
  
  if int(opts.batch) > 1:
    for x in range(0, int(opts.batch)):
      _generate(opts)
  else:
      _generate(opts)

def prettyprint(var):
  import pprint
  pp = pprint.PrettyPrinter()
  pp.pprint(var)
  sys.exit()

if __name__ == '__main__':
  opts = parse_args()
  # create API client with custom host, port
  api = webuiapi.WebUIApi(host=opts.host, port=opts.port)
  
  if opts.getmodel:
    model = api.util_get_current_model()
    print('Current model: %s' % model)
    sys.exit()

  if opts.setmodel:
    api.util_set_model(opts.setmodel)
    
  if opts.listmodels:
    models = api.util_get_model_names()
    prettyprint(models)

  if opts.listmodelsfull:
    models = api.get_sd_models()
    prettyprint(models)

  if opts.listsamplers:
    samplers = api.get_samplers()
    prettyprint(samplers)
  
  if opts.liststyles:
    styles = api.get_prompt_styles()
    prettyprint(styles)

  if opts.getoptions:
    options = api.get_options() 
    prettyprint(options)

  if opts.prompts:
    with open(opts.prompts) as prompts:
      for line in prompts:
        opts.prompt = line
        print(opts.prompt)
        generate(opts)
  else:
      generate(opts)
