# Interface with the ComfyUI API to create batches of videos
# Right now only working for WAN, requires ComfyScript
#
# TODO:
#   Add command line switches for all workflow settings
#   Add other workflows, import yaml, sys, wan, time
#   Add voice processing - sound effects, text to speech/cloning/sound effects
#  *Support other types of t2v and i2v. e.g., generate a tvshow for another scene.

import yaml, sys, time
from datetime import datetime as dt
import ollama, ollama
import wan

llm = ollama.Client('http://localhost:11434')
llm_model = 'ollama.com/library/deepseek-r1:8b'

data = yaml.load(open(sys.argv[1]),  Loader=yaml.CLoader)
wan_prompts = yaml.load(open('prompts.yaml'), Loader=yaml.CLoader)

scenario = data['name']
suffix = data['atmosphere']
scenes = data['scenes']

current = 1
pid = dt.now().strftime('%Y_%m_%d_%H_%M')

for scene in scenes:
  vid_prompt = '%s, %s' % (scene, suffix)
  print('Enhancing: ', vid_prompt[0:160], '...')
  response = llm.chat(model=llm_model, messages=[{'role':'system', 'content': wan_prompts['engineer']}, {'role': 'user', 'content': vid_prompt}])
  llm_prompt = response['message']['content']
  print('Queueing: ', llm_prompt[0:160], '...')
  wan.run(llm_prompt, 'WAN/%s-%s/scene%02d' % (scenario, pid, current))
  current = current + 1
  print('\n')
