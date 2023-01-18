rem place in webui directory
rem if pip produces errors in red text, re-run this script

call venv\scripts\activate.bat
git pull
pip install -U -r requirements.txt
