# nano-gpt-lightning
Lightning implementation of nano-gpt

# Get started
.. code:: bash
   git clone git@github.com:radulescupetru/nano-gpt-lightning.git
.. code:: bash
   conda create -n nano-gpt-lightning python=3.9
   conda activate nano-gpt-lightning
.. code:: bash
   cd nano-gpt-lightning
   pip install -r requirements.txt
.. code:: bash
   wandb login
   // Use wandb credentials to login
.. code:: bash
   python scripts/run.py fit --config /path_to/nano-gpt-lightning/nano_gpt_lightning/configs/datamodule_config.yaml --config /path_to/nano-gpt-lightning/nano_gpt_lightning/configs/model_config.yaml
