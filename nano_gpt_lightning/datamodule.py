import pytorch_lightning as pl
from torch.utils.data import DataLoader
from nano_gpt_lightning.data.shakespeare_dataset import ShakespeareDataset


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, data_url: str):
        super().__init__()
        self.data_dir = data_dir
        self.data_url = data_url

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
            ShakespeareDataset(self.data_dir, self.data_url, train=True),
            batch_size=self.trainer.model.args['core']['batch_size'],
            num_workers=self.trainer.model.args['core']['num_workers']
        )

    def val_dataloader(self):
        return DataLoader(ShakespeareDataset(self.data_dir, self.data_url, train=False),
                          batch_size=self.trainer.model.args['core']['batch_size'])

    def test_dataloader(self):
        return DataLoader(ShakespeareDataset(self.data_dir, self.data_url, train=False),
                          batch_size=self.trainer.model.args['core']['batch_size'])

    def predict_dataloader(self):
        return DataLoader(ShakespeareDataset(self.data_dir, self.data_url, train=False),
                          batch_size=self.trainer.model.args['core']['batch_size'])
