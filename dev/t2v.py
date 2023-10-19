import sys, os, re, random, time, string
import requests, base64, json, argparse
from urllib.parse import urlencode

# t2v generator v0.1: Generate videos from text prompts
# Uses Automatic1111 WebUI with API access and the text2video extension.
# In settings/text2video, select "All" under 'Keep model in VRAM between runs'
# Using the proper resolution for each model helps, good luck!
#
# TODO: Check request return, more error checking
#       Image/video to video (post request failing)
#       Upscaling
#       add all options from https://github.com/kabachuha/sd-webui-text2video/blob/main/scripts/api_t2v.py

def generate(prompt, uri,
        n_prompt=None,
        width=256,
        height=256,
        cfg=17,
        frames=24,
        fps=15,
        model='t2v',
        steps=30,
        seed=-1,
        sampler='DDIM_Gaussian',
        add_meta=True,
        rm_tmp=True,
        save_vid=True,
        img2img=None,
        continue_vid=False,
        inpainting_frames=1,
        strength=1,
        vid2vid=None,
        vid2vid_start=0):
  
  headers = { 'accept': 'application/json', 'Content-Type': 'application/json' }
  
  data = {'prompt': prompt, 'n_prompt': n_prompt,  'cfg': cfg, 'width': width, 'height': height, 'frames': frames,
          'fps': fps, 'steps': steps, 'seed': seed, 'model': model, 'strength': strength, 'sampler': sampler}

  if img2img:
    with open(img2img, 'rb') as f:
      img = f.read()
    
    img_b64 = base64.b64encode(img).decode('utf8')
    
    data['inpainting_image'] = img
    data['inpainting_frames'] = 1
  
  if vid2vid:
    with open(vid2vid, 'rb') as f:
      vid = f.read()
      vid = base64.b64encode(vid).decode('utf8')
    
    data['do_vid2vid'] = True
    data['vid2vid_input'] = vid
    data['vid2vid_startFrame'] = vid2vid_start

  url = '%s/t2v/run?%s' % (uri, urlencode(data))

  print('Processing "%s": ' % (prompt), end='')
  
  start_t = time.time()
  req = requests.post(url, headers=headers)
  end_t = time.time()
  
  print('%d secs' % (end_t - start_t))
  
  try:
    js = json.loads(req.content)
    s = js['mp4s'][0].replace('data:video/mp4;base64,','').encode()
    vid = base64.decodebytes(s)
  except Exception as e:
    print('ERROR: Loading response JSON: ', e)
    return {}

  # reading the seed from the generation would be nice
  r = random.randint(1,999999)
  
  if save_vid:
    outfile = '%s_%s.mp4' % (sanitize_filename_part(prompt), r)
    #outfile_tmp = outfile.replace('.mp4','_out.mp4')
  
    with open(outfile, 'wb') as outp:
      outp.write(vid)

#    if add_meta:
#      ff = 'ffmpeg -v 0 -i %s -preset fast -c:v libx264 -metadata comment="%s" -metadata artist="Stable Diffusion Text2Video" %s'  % (outfile_tmp,
#         ' '.join(['%s: %s' % (x, data[x]) for x in data.keys()]), outfile)
#    
#      os.system(ff)
    
    #if rm_tmp:
    #  os.unlink(outfile_tmp)
  else:
    return vid

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='t2v',
                                   description='Text 2 Video API helper',
                                   epilog='http://github.com/terbo/sdscripts')
  
  parser.add_argument('-p','--prompt', help='Text prompt')
  parser.add_argument('--api', help='API URI')
  parser.add_argument('-W','--width', help='Video width', default=256)
  parser.add_argument('-H','--height', help='Video height', default=256)
  parser.add_argument('--fps', help='Video fps', default=15)
  parser.add_argument('--frames', help='Total frames to render', default=24)
  parser.add_argument('-n','--negative', help='Negative prompt')
  parser.add_argument('-m','--model', help='ModelScope model', default='t2v')
  parser.add_argument('-s','--steps', help='Inference steps', default=30)
  parser.add_argument('-S','--seed', help='Seed (-1 for random)', default=-1)
  parser.add_argument('--sampler', help='Sampler', default='DDIM_Gaussian')
  parser.add_argument('-c','--cfg', help='CFG value', default=17)
  parser.add_argument('-f','--prompts', help='Text file containing prompts')
  
  opts = parser.parse_args()
  
  if opts.prompts:
    with open(opts.prompts) as fp:
      prompts = fp.read().splitlines()

    for prompt in prompts:
      generate(prompt, opts.api, n_prompt=opts.negative, width=opts.width, height=opts.height,
             cfg=opts.cfg, frames=opts.frames, fps=opts.fps, sampler=opts.sampler,
             seed=opts.seed, steps=opts.steps, model=opts.model)
  elif opts.prompt and opts.api:
    generate(opts.prompt, opts.api, n_prompt=opts.negative, width=opts.width, height=opts.height,
             cfg=opts.cfg, frames=opts.frames, fps=opts.fps, sampler=opts.sampler,
             seed=opts.seed, steps=opts.steps, model=opts.model)
  else: 
    parser.print_help()
