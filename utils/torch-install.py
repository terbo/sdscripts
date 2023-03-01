import os
# add linux/win32/macos/cpu versions? non-programatically??

commands = { 
            'pip/cu116':
              'pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116',
            'pip/cu117':
              'pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117',
            'pip/orig':
              'pip install -U torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio --extra-index-url https://download.pytorch.org/whl/cu117',
            'conda/117':
              'conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia',
            'conda/116':
              'conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia'
     }

i = 1
cmds = []

print('Pick a CUDA version and install method:\n')

for c in sorted(commands.keys()):
  cmds.append(c)
  print(f'\t [{i}]\t{c}')
  i += 1

r = [ x for x in range(1,len(commands.keys())+1) ]
inp = int(input(f'{r} > '))

command = commands[cmds[inp-1]]

if input(f'Execute "{command}"? [y/N] ') in ['y','Y']:
    os.system(command)
