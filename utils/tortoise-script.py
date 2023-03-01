#!env python
# read text files with embedded commands for tortoise tts

# requires tortoise-tts, nltk, and np
'''
CLI? GUI?

Change voices, speeds, temperments, show list of voices, options

use configparser and sections/comments to direct voice
preprocess to remove all things similar to commands/comments

[title? filename? voice? command?]
text

[train-grace]
; casual

voice
seed
candidates

TODO: Generate silence

Send API request:
  script
  format
  api key
'''

import os, sys, time
import argparse
import torchaudio
from pprint import pprint
from datetime import datetime as dt
from nltk.tokenize import sent_tokenize
#from pydub import AudioSegment
from np import random

def parse_args():
  parser = argparse.ArgumentParser(prog='tortoise-script',
                                    description='Script Tortoise TTS Content',
                                    epilog='http://github.com/terbo/tortoise-tts-script/')
  parser.add_argument('-c', '--candidates', default=1, help='how many candidates per generation for comparison')
  parser.add_argument('-s', '--seed', help='input fixed seed or use random')
  parser.add_argument('-r', '--rate', default=24000, help='sample rate, default 24000')
  parser.add_argument('-v', '--voice', default=None, help='voice to use')
  parser.add_argument('-q', '--quality', default='fast', help='quality to encode [ultra_fast, fast, standard, high_quality]')
  parser.add_argument('-f', '--file', help='script to process')
  parser.add_argument('-l', '--listvoices', action='store_true', help='list all voices')
  parser.add_argument('-i', '--info', action='store_true', help='show environment information')

  return parser.parse_args()

#try:
voices = os.listdir('tortoise/voices')
opts = parse_args()

if opts.info:
  import torch
  print(f'Python: {sys.version.split(" ")[0]}')
  print(f'Torch: {torch.__version__}')
  print(f'Torchaudio: {torchaudio.version.__version__}')
  print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda_version}')
  print(f'Voices: {voices}')
  sys.exit()

if opts.listvoices:
  pprint(voices)
  sys.exit()

voice = opts.voice # or random_voice
presets = ["ultra_fast", "fast", "standard", "high_quality"]
assert opts.quality in presets, f'invalid quality: {opts.quality}, valid are {presets}'
preset = opts.quality

with open(opts.file) as infile:
  lines = infile.readlines()

if not opts.seed:
  opts.seed = random.randint(99999999)

print(len(lines), 'lines read')

sents =  sent_tokenize(' '.join(lines))
samples = []


out = []
#outputs = AudioSegment.silent(duration=500, frame_rate=opts.rate)
outfiles = []

n = 0
voice = opts.voice

for line in lines:
  if line.startswith('#'):
    cmd = line[1:].split(' ')[1]
    cmd = cmd.strip()
    try:
      args = line[1:].split(' ')[2]
      args = args.strip()
    except:
      pass
    if cmd == 'voice':
      voice = args
      out.append({'voice': voice, 'text': []})
    elif cmd == 'emotion':
      line = f'[I am really {args.lower(),}]  {line}'
    elif cmd == 'silence':
      out[n]['silence'] = float(args)
    elif cmd == 'text':
      lines=[]
      text = open(args).readlines()
      for line in text: lines.append(line.strip())
      print(len(text), ' lines read')
      out.append({'voice': voice, 'text': lines})
    elif cmd == 'quality':
      out[n]['quality'] = args
    elif cmd == 'highpass':
      out[n]['compress'] = args
    elif cmd == 'highpass':
      out[n]['highpass'] = args
    elif cmd == 'lowpass':
      out[n]['lowpass'] = args
    elif cmd == 'pan':
      out[n]['pan'] = args
    elif cmd == 'eq':
      out[n]['eq'] = args
  else:
    if len(line) <= 0:
      continue
    elif(len(line) >= 128):
      n += 1
      out.append({'voice': voice, 'text': []})
    try:
      out[n]['text'].append(line)
    except:
      n -= 1
      out[n]['text'].append(line)
    n += 1

#pprint(out)

print(f'{opts.candidates} candidates, {opts.seed} seed, {opts.voice} voice, {opts.quality} quality')
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
tts = TextToSpeech()

for segment in out:
  print(segment['text'])
  voice = segment['voice']
  voice_samples, conditioning_latents = load_voices([voice])
  text = ' '.join(segment['text'])
  start_time = time.time()
  gen, dbg = tts.tts_with_preset(
    text,
    voice_samples=voice_samples,
    conditioning_latents=conditioning_latents,
    preset=opts.quality,
    use_deterministic_seed=opts.seed,
    return_deterministic_state=True,
    k=int(opts.candidates),
  )
  print(
    f"{dt.now()} | Voice: {opts.voice} | Text: {text} | Quality: {opts.quality} | Time Taken (s): {time.time()-start_time} | Seed: {opts.seed}\n")
  if isinstance(gen, list):
    n=0
    for j, g in enumerate(gen):
      n+=1
      outfiles.append(f"{voice}-{opts.quality}_{opts.seed}_{n}.wav")
      torchaudio.save(outfiles[-1], g.squeeze(0).cpu(), opts.rate)
  else:
    try:
      outfiles.append(f"{opts.voice}-{start_time}_{opts.seed}.wav")
      torchaudio.save(outfiles[-1], gen.squeeze(0).cpu(), opts.rate)
    except Exception as e:
      print(e)
   
  #if 'silence' in segment.keys():
  #    outputs += AudioSegment.silent(duration=segment['silence'], frame_rate=opts.rate)

  try:
    pass
    #outputs.export(f'{start_time}-{opts.voice}-{opts.seed}.wav', format='wav')
  except:
    #files = AudioSegment.silent(duration=500, frame_rate=opts.rate)
    print('Done. Combining files...')
    #for outfile in outfiles:
      #files += AudioSegment.from_file(outfile)
    #files.export("{opts.voice}-{start_time}_{opts.seed}.wav", format='wav')
    pass

#except Exception as e:
#  print(f'\nError: {e}')

'''
[{'text': ["i've got you now my pretty!"], 'voice': 'freeman'},
 {'silence': 350.0, 'text': ['this is so nice here!'], 'voice': 'rainbow'},
 {'text': ['I shall do with you as I want!', 'You are all mine!'],
  'voice': 'freeman'},
 {'silence': 500.0,
  'text': ["oh no, gee mister don't hurt me!"],
  'voice': 'rainbow'},
 {'text': ["I won't hurt you little girl....",
           'Far from that ...',
           'hahahaha',
           'hahaha',
           'hahahahahaha'],
  'voice': 'freeman'},
 {'text': ['you see, my pretty',
           'I am going to cook you up in a stew fit for a king!',
           'I have all of the ingredients ready, and I am absolutely starving!',
           'come to me!',
           'hahahahaha'],
  'voice': 'freeman'}]
'''
