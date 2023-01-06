from pytorch_lightning.cli import LightningCLI

from nano_gpt_lightning.datamodule import GPTDataModule
from nano_gpt_lightning.trainmodule import LitNanoGPT

if __name__ == '__main__':
    cli = LightningCLI(model_class=LitNanoGPT, datamodule_class=GPTDataModule)
