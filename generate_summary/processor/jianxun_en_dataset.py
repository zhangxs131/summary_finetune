import torch
import json
import pandas as pd


class Jianxun_EN_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path,prefix=""):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_source_length = 1024
        self.max_target_length = 256
        self.prefix = prefix
        self.padding="max_length"

        if data_path.split('.')[-1]=='csv':
            df=pd.read_csv(data_path)
        elif data_path.split('.')[-1] == 'json':
            with open(data_path, 'r', encoding='utf-8') as f:
                content = json.loads(f.read())

            self.data = []
            for i, data in enumerate(content):
                self.data.append((data['英文整编'], data['原文']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        inputs = line[1]
        targets = line[0]
        inputs = self.prefix + inputs

        # Tokenize Input
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)


        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" :
            labels["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

