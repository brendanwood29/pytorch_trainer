# Pytorch Trainer
This project was inspired by pytorch-lightning. I wanted to create my own version so that I could avoid creating reused code and better understand the inner workings of the lightning training pipeline.

# Installation
1. Install `uv` according to the system requirements, linked [here](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv venv` to initalize a virtual environment.
3. Install `pytorch` accoring to your system environment using the command:
```bash
uv pip install torch==<version> --index-url --index-url https://download.pytorch.org/whl/cu<version>
```
4. Clone this repository with `git clone https://github.com/brendanwood29/pytorch_trainer.git`
5. Add the project to the current environment with `uv add --editable ./pytorch_trainer`
6. Run `uv sync` to install remaining dependencies.

# Contents
## abstracts
Contains basic code skeletons for inspiration, if you wish to overwrite anything.
## Defaults
Default configured losses, optims, schedulers, and early stopper. You should not need to make your own, these are dynamic and can be edited.