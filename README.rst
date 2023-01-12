================
Lightning implementation of nano-gpt
================

Getting started
-----------

Installation

Clone the repo

.. code:: bash

   git clone git@github.com:radulescupetru/nano-gpt-lightning.git

Create conda env

.. code:: bash

   conda create -n nano-gpt-lightning python=3.9
   conda activate nano-gpt-lightning
   
Install requirements

.. code:: bash

   cd nano-gpt-lightning
   pip install -r requirements.txt

Login to wandb using your credentials

.. code:: bash

   wandb login

Sample fit command

.. code:: bash

   python scripts/run.py fit --config /path_to/nano-gpt-lightning/nano_gpt_lightning/configs/datamodule_config.yaml --config /path_to/nano-gpt-lightning/nano_gpt_lightning/configs/model_config.yaml
