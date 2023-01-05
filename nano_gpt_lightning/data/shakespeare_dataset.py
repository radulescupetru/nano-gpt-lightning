import pathlib

import numpy as np
import requests
import tiktoken
import torch.utils.data
from torch.utils.data.dataset import T_co


class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, data_url: str, train: bool = False, **kwargs):
        if not pathlib.Path(data_dir, "train.bin").exists():
            ShakespeareDataset.prepare(data_dir, data_url)
        self.train = train
        self.data = np.memmap(pathlib.Path(data_dir, "train.bin" if self.train else "val.bin"))
        self.block_size = kwargs.get("block_size", 1024)

    def __getitem__(self, index) -> T_co:

        x = torch.from_numpy((self.data[index:index + self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[index + 1:index + 1 + self.block_size]).astype(np.int64))
        return x, y

    def __len__(self):
        return len(self.data)

    @staticmethod
    def prepare(data_dir, data_url):
        # Handle the case when the data was not downloaded locally
        if not pathlib.Path(data_dir, 'input.txt').exists():
            with open(pathlib.Path(data_dir, 'input.txt'), 'w') as file_writer_stream:
                file_writer_stream.write(requests.get(data_url).text)
        # Data is available locally
        with open(pathlib.Path(data_dir, 'input.txt'), 'r') as file_reader_stream:
            data = file_reader_stream.read()
            train_data = data[:int(0.9 * len(data))]
            val_data = data[int(0.9 * len(data)):]
        # Define token encoder
        encoder = tiktoken.get_encoding("gpt2")
        # Encode train and validation data
        train_ids = encoder.encode_ordinary(train_data)
        val_ids = encoder.encode_ordinary(val_data)

        # Save encodings to disk
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(pathlib.Path(data_dir, 'train.bin').as_posix())
        val_ids.tofile(pathlib.Path(data_dir, 'val.bin').as_posix())
