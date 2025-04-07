# Install

## Install Python 3.10

[Python download page](https://www.python.org/downloads/)

### Ubuntu
`sudo apt-get install python3.10`

### Ubuntu w/o sudo access
`curl https://pyenv.run | bash`
Follow instructions for getting pyenv to load automatically. Append the following to the bottom of .bashrc :
`export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"`
Open a new terminal window.
`pyenv install 3.10`
Set version of python 3.10 as global python version or local version:
`pyenv global 3.10.xx`
OR
`pyenv local 3.10.xx`

## Create a virtual environment

`python3.10 -m venv .venv`

## Activate your virtual environment

`source .venv/bin/activate`

## Clone the repository

`git clone https://github.com/Union-College-Computer-Science/exp_framework.git`

## Install requirements

`pip3 install -r requirements.txt`
