@echo off

rem Version 0.5
rem
rem My automatic1111 launch batch file, from webui-user.bat
rem Uses variables for common settings, specifies alternate directories for models,
rem and allows for adding flags from the command line
rem
rem Should I copy/paste the help descriptions as comments?
rem
rem Reference: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings

set PYTHON=
set GIT=
set VENV_DIR=

rem UI Settings

set LISTEN=--listen --port 1111
set STYLES=--styles-file g:\\sd\\www\\styles.csv
set THEME=--theme dark

rem Runtime setttings
set API=--api
set NOCUDA=--skip-torch-cuda-test
set XFORMERS=--xformers
set XFORMERS_INSTALL=--reinstall-xformers

rem Memory Settings
set PRECISION=--no-half
set MEMORY_FLAGS=--medvram --opt-split-attention
set AUTOCAST=--precision autocast

rem Model paths
set MODELS=--ckpt-dir g:\\sd\\models
rem set HUGGINGFACE=g:\\sd\\huggingface

set CODEFORMER=--codeformer-models-path g:\\sd\\models\\util
set GFPGAN=--gfpgan-models-path g:\\sd\\models\\util
set ESRGAN=--esrgan-models-path g:\\sd\\models\\util
set EMBEDDINGS=--embeddings-dir g:\\sd\\models\\embeddings

set COMMANDLINE_ARGS=%MODELS% %STYLES% %EMBEDDINGS% %XFORMERS% %CODEFORMER% %GFPGAN% %ESRGAN% %API% %*

rem TODO
rem --share
rem --gradio-auth
rem --gradio-img2img-tool
rem --gradio-debug
rem --api
rem --nowebui
rem --ui-config-file
rem --ui-settings-file
rem --config
rem --vae-path
rem --freeze-settings
rem Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)
rem --device-id	DEVICE_ID	

call webui.bat
