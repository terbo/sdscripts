#!env python
# show python, torch, and cuda information

import os, sys, time

print(f'Python: {sys.version.split(" ")[0]}/{sys.platform}')

try:
  import torch
  print(f'Torch: {torch.__version__}')
  print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda_version}')
  try:
    print(f'Torchaudio: {torchaudio.version.__version__}')
  except:
    print('No torchaudio detected')
except:
  print('No torch detected')
