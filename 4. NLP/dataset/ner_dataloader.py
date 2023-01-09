from torch.utils.data import Dataset
import typing
from transformers import AutoTokenizer, MLukeTokenizer
import numpy as np
import torch

class NerDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 model_name_or_path: str,
                 tags_list: typing.List[str],
                 max_seq_length: int = 128,
                 label_all_tokens: bool = False):

        self.max_seq_length = max_seq_length
        self.label_all_tokens = label_all_tokens

        if 'mluke' in model_name_or_path:
            self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tag2id = {}
        for i in range(len(tags_list)):
            self.tag2id[tags_list[i]] = i

        self.dataset = []
        sen_dataset = self.read_data(dataset_path=dataset_path)
        for i in range(len(sen_dataset)):
            conll_format = sen_dataset[i]
            tokenized_inputs = self._tokenize_and_align_labels(data_point=conll_format)
            self.dataset.append(tokenized_inputs)

    @staticmethod
    def read_data(dataset_path: str) -> list:
        dataset = []
        with open(dataset_path, 'r') as file:
            lines = file.readlines()

        lines = [line.strip('\n').replace('\t', ' ').replace('B-MISCELLANEOUS', 'O').replace('I-MISCELLANEOUS', 'O')
                 for line in lines]
        break_idxs = list(np.where(np.array(lines) == '')[0])

        for i in range(len(break_idxs) - 1):
            start_idx = break_idxs[i]
            end_idx = break_idxs[i + 1]

            if start_idx != 0:
                start_idx += 1

            dataset.append(lines[start_idx: end_idx])

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item: dict = self.dataset[idx]

        return item

    def _tokenize_and_align_labels(self, data_point: typing.List[str]):
        word_list = [word.split(' ')[0] for word in data_point]
        label_list = [word.split(' ')[-1] for word in data_point]

        text = ' '.join(word_list)
        tokenized_inputs = self.tokenizer(text,
                                          truncation=True,
                                          is_split_into_words=False,
                                          max_length=self.max_seq_length,
                                          padding='max_length')
        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []
        count = 0

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(self.tag2id[label_list[word_idx]])
                except:
                    count += 1
                    label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                try:
                    label_ids.append(self.tag2id[label_list[word_idx]] if self.label_all_tokens else -100)
                except:
                    count += 1
                    label_ids.append(-100)
            previous_word_idx = word_idx

        # print('Num error tags: {}'.format(count))
        tokenized_inputs["labels"] = torch.LongTensor(label_ids)
        tokenized_inputs['input_ids'] = torch.LongTensor(tokenized_inputs['input_ids'])
        tokenized_inputs['attention_mask'] = torch.LongTensor(tokenized_inputs['attention_mask'])
        return tokenized_inputs