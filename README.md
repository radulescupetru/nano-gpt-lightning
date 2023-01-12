# nano-gpt-lightning
Lightning implementation of nano-gpt

# Get started
git clone git@github.com:radulescupetru/nano-gpt-lightning.git

conda create -n nano-gpt-lightning python=3.9
conda activate nano-gpt-lightning

cd nano-gpt-lightning
pip install -r requirements.txt

wandb login
 - Use wandb credentials to login

python scripts/run.py fit --config /path_to/nano-gpt-lightning/nano_gpt_lightning/configs/datamodule_config.yaml --config /path_to/nano-gpt-lightning/nano_gpt_lightning/configs/model_config.yaml
