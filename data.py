import random
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


logger = logging.getLogger(__name__)


class RawDataProvider(object):
    
    def __init__(self, train_path, test_path):
        self.train_df = self.load_csv(train_path)
        self.test_df = self.load_csv(test_path)
        
        self.all_idiom = self.__get_all_idiom([self.train_df, self.test_df])
    
    def load_csv(self, path="./data/train.csv"):
        return pd.read_csv(path, sep="\t")
    
    def __get_all_idiom(self, dfs):
        idiom_set = set()
        for df in dfs:
            for i in df["candidate"]:
                idiom_set |= set(i)
    
        return list(idiom_set)
    
    def build_data_samples_for_single(
        self, single_df, expected_extend_neg, unique=True, max_neg_count=99,
    ):
            
        def get_global_negative_sample(expected):
            # TODO: get hard neg with cos similarity
            samples = random.sample(self.all_idiom, expected)
            return samples
    
        text = single_df["text"]
        candidate = eval(single_df["candidate"])
        label = single_df["label"]

        if expected_extend_neg > 0:
            # logger.info(f"进行额外负采样，长度:{expected_extend_neg}")
            extend_negative = get_global_negative_sample(expected_extend_neg)
            candidate.extend(extend_negative)
            random.shuffle(candidate)
            
        neg_count = 0
        samples = []
        if unique:
            candidate = set(candidate)
            
        for s in candidate:
            if s == label:
                samples.append([text, s, 1])
            else:
                if neg_count < max_neg_count:
                    samples.append([text, s, 0])
                    neg_count += 1

        return samples

    def get_all_samples(self, df, expected_extend_neg, max_neg_count):
        samples = []
        
        for i in range(len(df)):
            samples.extend(
                self.build_data_samples_for_single(
                    df.iloc[i], expected_extend_neg, max_neg_count
                )
            )

        return samples

    def get_train_valid_test_samples(
        self, expected_extend_neg, max_neg_count, valid_rate=0.1,
    ):
        train_samples = self.get_all_samples(
            self.train_df, expected_extend_neg, max_neg_count,
        )
        test_samples = self.get_all_samples(
            self.test_df, expected_extend_neg, max_neg_count,
        )
        
        valid_len = int(len(train_samples) * valid_rate)
        
        valid_samples = train_samples[:valid_len]
        train_samples = train_samples[valid_len:]
        
        logger.info(
            f"train:{len(train_samples)}, "
            f"valid:{len(valid_samples)}, "
            f"test:{len(test_samples)}"
        )
        
        return train_samples, valid_samples, test_samples


class IdiomDataset(Dataset):
    
    def __init__(self, samples, tokenizer, max_length):
        """dataset.

        Args:
            samples (list): [text, idiom, label]
            tokenizer (tokenizer): BertTokenizer
            max_length (int): max input length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.build(self.samples[item])
    
    def build(self, sample):
        data = self.tokenizer.encode_plus(
            *sample[:2], max_length=self.max_length, truncation=True, 
            return_tensors="pt",
        )
        
        data["label"] = torch.tensor(sample[-1]).long()
        data["position_ids"] = torch.arange(0, data["input_ids"].shape[-1])
        
        return data
        

def collate_fn(batch):
    input_ids = pad_sequence(
        [i["input_ids"].squeeze(0) for i in batch], batch_first=True,
    )
    token_type_ids = pad_sequence(
        [i["token_type_ids"].squeeze(0) for i in batch], batch_first=True,
    )
    attention_mask = pad_sequence(
        [i["attention_mask"].squeeze(0) for i in batch], batch_first=True,
    )
    position_ids = pad_sequence(
        [i["position_ids"].squeeze(0) for i in batch], batch_first=True,
    )
    
    label = torch.stack([i["label"] for i in batch], dim=0)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "label": label,
    }


if __name__ == "__main__":
    rdp = RawDataProvider("./data/train.csv", "./data/test.csv")

    a, b, c = rdp.get_train_valid_test_samples(expected_extend_neg=2)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    trainset = IdiomDataset(a, tokenizer=tokenizer, max_length=512)
    dl = DataLoader(trainset, batch_size=3, collate_fn=collate_fn)
    
    for i in dl:
        print(i)
        break
