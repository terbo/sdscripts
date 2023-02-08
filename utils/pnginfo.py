import sys, glob
from PIL import Image
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='pnginfo',
                                   description='Extract generation parameters from images created with Automatic 1111')
  parser.add_argument('-prompt', action='store_true', help='Display only prompt')
  parser.add_argument('-params', action='store_true', help='Display only parameters')
  parser.add_argument('-i', help='Select images, globs valid')
  
  opts = parser.parse_args()

  if len(opts.i) < 1:
    parser.print_help()
    sys.exit()
  
  for filename in glob.glob(opts.i):
    im = Image.open(filename)
    im.load()  # Needed only for .png EXIF data (see citation above)
    
    try:
      params = im.info['parameters'].split('\n')
      prompt = params[0]
      details = params[1]

      if opts.prompt:
        print(f'{filename}: {prompt}')
      elif opts.params:
        print(f'{filename}: {details}')
      else:
        print(f'Image: {filename}\nPrompt: {prompt}\n{details}')

    except:
      print(f'{filename}: no info')
