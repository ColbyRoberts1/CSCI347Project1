## Getting started
This repo uses python virtual environments to manage dependencies. This simplifies dependency management and ensures all group members can run the code in the same environment. 

### Steps
1. Clone the repo and enter
```bash
git clone https://github.com/ColbyRoberts1/CSCI347Project1.git

cd CSCI347Project1
```

2. Activate the virtual environment

Linux and macOS
```bash
source ./bin/activate
```
Windows
```bash
# In cmd.exe
venv\Scripts\activate.bat
# In PowerShell
venv\Scripts\Activate.ps1
```

### Deactivate venv
Once you have finished working on your project, it’s a good habit to deactivate its venv. By deactivating, you leave the virtual environment. Without deactivating your venv, all other Python code you execute, even if it is outside your project directory, will also run inside the venv.

Luckily, deactivating your virtual environment couldn’t be simpler. Just enter the following. It works the same on all operating systems.

```bash
deactivate
```