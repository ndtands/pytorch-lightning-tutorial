from pytorch_lightning.core import LightningDataModule
import typing
import os
from torch.utils.data import DataLoader
from dataset.ner_dataloader import NerDataset
from configs import PATH_DATASET

class NERDataModule(LightningDataModule):

    def __init__(self,
                 model_name_or_path: str,
                 dataset_version: str,
                 tags_list: typing.List[str],
                 label_all_tokens: bool = False,
                 max_seq_length: int = 128,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 test_size: float = 0.1,
                 val_size: float = 0.2,
                 **kwargs):
        super().__init__()

        self.val_data = None
        self.train_data = None
        self.dataset_df = None
        self.test_data = None
        self.save_hyperparameters(ignore=['model_name_or_path', 'tags_list', 'test_size', 'val_size'])

        self.model_name_or_path = model_name_or_path
        self.dataset_version = dataset_version
        self.tags_list = tags_list
        self.num_labels = len(tags_list)
        self.label_all_tokens = label_all_tokens

        self.val_size = val_size
        self.test_size = test_size

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def setup(self, stage: typing.Optional[str] = None):

        self.train_data = NerDataset(
            dataset_path=os.path.join(PATH_DATASET, self.dataset_version, 'train_data.txt'),
            model_name_or_path=self.model_name_or_path,
            tags_list=self.tags_list,
            max_seq_length=self.max_seq_length,
            label_all_tokens=self.label_all_tokens)

        self.val_data = NerDataset(
            dataset_path=os.path.join(PATH_DATASET, self.dataset_version, 'val_data.txt'),
            model_name_or_path=self.model_name_or_path,
            tags_list=self.tags_list,
            max_seq_length=self.max_seq_length,
            label_all_tokens=self.label_all_tokens)

        self.test_data = NerDataset(
            dataset_path=os.path.join(PATH_DATASET, self.dataset_version, 'test_data.txt'),
            model_name_or_path=self.model_name_or_path,
            tags_list=self.tags_list,
            max_seq_length=self.max_seq_length,
            label_all_tokens=self.label_all_tokens)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.eval_batch_size)

    def predict_dataloader(self):
        pass