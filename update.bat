rem place in webui directory
rem if pip produces errors in red text, re-run this script
rem additionally you may need to update pip, you can do that
rem by running the following line before this script (without the "rem" part):
rem pip install --upgrade pip

call venv\scripts\activate.bat
git pull
pip install -U -r requirements.txt

pause
