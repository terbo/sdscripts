"""
Using Automatic 1111 WebUI, webuiapi and PIL,
generate an image from the command line

CSV format:
prompt,negative_prompt,width,height,steps,seed,cfg,sampler,model,style

Enclose strings in quotes.

Requires https://github.com/AUTOMATIC1111/stable-diffusion-webui with '--api' command line flag
Based on https://github.com/mix1009/sdwebuiapi

TODO: tiling, img2img, loopback, scripts, hires fix, upscaling, face restore
      if making a batch, create a collage of all images, maybe
      generate more unique output files

BUGS: If the server crashes (or in my case the ssh tunnel times out..)
      the API blocks.
"""

import argparse, string, random
import re, sys
import webuiapi
from PIL import Image

defaults = {}

def read_yaml(yaml_filename='generate.yaml'):
  if len(sys.argv) > 3 and sys.argv[1] == '--yaml':
    yaml_filename = sys.argv[2]
  
  try:
    import yaml
    
    with open(yaml_filename, "r") as ymlfile:
      yamlcfg = yaml.safe_load(ymlfile)
      
      defaults['width'] = yamlcfg['width']
      defaults['height'] = yamlcfg['height']
      defaults['negative_prompt'] = yamlcfg['negative_prompt']
      defaults['steps'] = yamlcfg['steps']
      defaults['seed'] = yamlcfg['seed']
      defaults['cfg'] = yamlcfg['cfg']
      defaults['sampler'] = yamlcfg['sampler']
      defaults['model'] = yamlcfg['model']
      defaults['batch'] = yamlcfg['batch']
      defaults['styles'] = yamlcfg['styles']
      defaults['host'] = yamlcfg['host']
      defaults['port'] = yamlcfg['port']
      defaults['output_dir'] = yamlcfg['output_dir']
      defaults['seed_base'] = yamlcfg['seed_base']

  except Exception as e:
    print('Loading yaml: ', e, sys.argv)
    
    defaults['width'] = 512
    defaults['height'] = 512
    defaults['negative_prompt'] = ''
    defaults['steps'] = 20
    defaults['seed'] = 0
    defaults['cfg'] = 7
    defaults['sampler'] = 'DPM++ 2S a Karras'
    defaults['model'] = 'v1-5-pruned-emaonly.ckpt'
    defaults['batch'] = 1
    defaults['styles'] = ''
    defaults['host'] = 'localhost'
    defaults['port'] = 7860
    defaults['output_dir'] = 'images'
    defaults['seed_base'] = 1e15  # 1000000000000000

def parse_args():
    parser = argparse.ArgumentParser(prog='generate',
                                   description='Partial command line interface for Automatic 1111 Web UI',
                                   epilog='http://github.com/terbo/sdscripts')
    
    parser.add_argument('--prompt', help='prompt to generate image from')
    parser.add_argument('--prompts', help='read prompts from a text file')
    parser.add_argument('--promptcsv', help='read prompts and generation options from a csv file')

    parser.add_argument('--width', help='Image width', default=defaults['width'])
    parser.add_argument('--height', help='Image height', default=defaults['height'])
    
    parser.add_argument('--negative_prompt', help='negative prompt', default=defaults['negative_prompt'])
    parser.add_argument('--steps', help='sampling steps', default=defaults['steps'])
    parser.add_argument('--seed', help='default seed', default=defaults['seed'])
    parser.add_argument('--seedbase', help='default seed', default=defaults['seed_base'])
    parser.add_argument('--cfg', help='CFG scale (0-30)', default=defaults['cfg'])
    parser.add_argument('--sampler', help='sampler to use', default=defaults['sampler'])
    parser.add_argument('--styles', help='comma separated list of styles to apply, use --liststyles to view styles', default=defaults['styles'])

    parser.add_argument('--batch', help='number of images to create', default=defaults['batch'])
    parser.add_argument('--output_dir', help='directory to output images to', default=defaults['output_dir'])
    
    parser.add_argument('--host', help='WebUI API Host', default=defaults['host'])
    parser.add_argument('--port', help='WebUI API Port', default=defaults['port'])
    
    parser.add_argument('--model', help='set default model in UI')
    parser.add_argument('--getmodel', action='store_true', help='show default model in UI')
    parser.add_argument('--listmodelsfull', action='store_true', help='list available models with all information')
    parser.add_argument('--listmodels', action='store_true', help='list available models (2)')
    
    parser.add_argument('--listsamplers', action='store_true', help='list available samplers')
    parser.add_argument('--liststyles', action='store_true', help='list available prompt styles')
   
    parser.add_argument('--yaml', help='select yaml configuration', default='generate.yaml')
    parser.add_argument('--getoptions', action='store_true', help='return current options')
    parser.add_argument('--showimg', action='store_true', help='show generated image')
    parser.add_argument('--saveinfo', action='store_true', help='save generation information')
    parser.add_argument('--showinfo', action='store_true', help='show generation information')
    
    parser.add_argument('--debug', action='store_true', help='show debug information')
    
    return (parser.parse_args(), parser.print_help)

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

def generate_image(options):
  if (not options.seed) or (options.seed == '0') or (options.seed == 0):
    options.seed = random.randint(1, int(float(defaults['seed_base'])))
    if options.debug:
      print('Seed: %d' % options.seed)
  
  prompt = "%s%s%s" % (options.prompt, ', ' if options.styles else '', (options.styles))

  res = api.txt2img(prompt=prompt,
                    negative_prompt=options.negative_prompt,
                    width=options.width,
                    height=options.height,
                    seed=options.seed,
                    styles=[], # concatenated with prompt, doesn't get applied here for some reason
                    cfg_scale=options.cfg,
                    sampler_index=options.sampler,
                    steps=options.steps,
                   )

  return (prompt, res)

def save_text_properties(options, output_file):
  txtfile = open(output_file, 'w')
  text = [options.prompt, "\n", "Steps: %d, Sampler: %s, CFG scale: %d, Seed: %d, Size: %dx%d, Model: %s" %
        (options.steps, options.sampler, options.cfg, options.seed, options.width, options.height, options.model)]
  
  txtfile.writelines(text)
  txtfile.close()
  if options.showinfo or options.debug:
    print(''.join(text))

def prettyprint(var):
  import pprint
  pp = pprint.PrettyPrinter()
  pp.pprint(var)
  sys.exit()

def read_csv(options):
  import csv

  with open(options.promptcsv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        options.prompt = row[0]
        options.negative_prompt = row[1] if row[1] else ''
        options.width = row[2] if row[2].isdigit() else defaults['width']
        options.height = row[3] if row[3].isdigit() else defaults['height']
        options.steps = row[4] if row[4].isdigit() else defaults['steps']
        options.seed = row[5] if row[5].isdigit() else defaults['seed']
        options.cfg = row[6] if row[6].isdigit() else defaults['cfg']
        options.sampler = row[7] if row[7] else defaults['sampler']
        options.model = row[8] if row[8] else defaults['model']
        options.styles = row[9] if row[9] else defaults['styles']
        
        line_count += 1
        current_model = api.util_get_current_model()
        
        if options.model != current_model:
          api.util_set_model(options.model)
        
        generate(options)
  
  sys.exit()

def generate(options):

  def _generate(options):
    (prompt, res) = generate_image(options)

    output_file = ('%d_%s' % (options.seed, prompt))
    image_name = options.output_dir + '\\' + sanitize_filename_part(output_file)
    res.image.save(image_name + '.png')
    if options.saveinfo:
      save_text_properties(options, image_name + '.txt')

    if options.showimg:
      image=Image.open(image_name + '.png')
      image.show()
  
  if int(options.batch) > 1:
    for x in range(0, int(options.batch)):
      options.seed = 0
      _generate(options)
  else:
      _generate(options)

if __name__ == '__main__':
  read_yaml()
  (opts, usage) = parse_args()
  #random.seed(31337)
  
  # create API client with custom host, port
  api = webuiapi.WebUIApi(host=opts.host, port=opts.port)
  
  if opts.getmodel:
    model = api.util_get_current_model()
    print('Current model: %s' % model)
    sys.exit()

  if opts.model:
    api.util_set_model(opts.model)
    
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

  if opts.promptcsv:
    read_csv(opts)

  if opts.prompts:
    with open(opts.prompts) as prompts:
      for line in prompts:
        opts.prompt = line
        generate(opts)
  
  elif opts.prompt != None:
      generate(opts)
  
  #elif len(sys.argv) > 2:
  #  opts.prompt = sys.argv[1]
  
  else:
    usage()
