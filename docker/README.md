# Docker notes

Training and serving are intended to run **inside Docker on GPU VMs**.\n
Local installs on macOS are optional.\n
If you want a local venv for linting/import checks:\n
```bash\n
python3 -m venv .venv\n
source .venv/bin/activate\n
python -m pip install -r requirements.txt\n
```\n
