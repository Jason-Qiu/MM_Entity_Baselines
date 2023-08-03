# All baselines

## Setup


```bash
conda create -n MMbaseline python=3.9

sudo apt install git-lfs

git clone https://github.com/catid/minigpt4
cd minigpt4

git lfs install
git submodule update --init --recursive

conda activate MMbaseline
pip install -r requirements.txt
pip install -e .
```

## Example Usage


```bash
run_all_baselines.ipynb
```
