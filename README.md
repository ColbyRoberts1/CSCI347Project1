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

3. Install dependencies

If you are using macOS you may need to replace `pip` with `pip3`.

```bash
pip install -r requirements.txt
```


### Deactivate venv
Once you have finished working on your project, it’s a good habit to deactivate its venv. By deactivating, you leave the virtual environment. Without deactivating your venv, all other Python code you execute, even if it is outside your project directory, will also run inside the venv.

Luckily, deactivating your virtual environment couldn’t be simpler. Just enter the following. It works the same on all operating systems.

```bash
deactivate
```

### Managing Dependencies
Ensure you are inside the virtual environment before modifying any dependencies to 
avoid corrupting your global installations. We will use pip to track dependencies. 
After adding or removing a dependency with pip, run the following command from the
project root directory to update the `requirements.txt` file.

```bash
pip freeze > requirements.txt
```
