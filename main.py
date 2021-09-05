import sys
import logging
sys.path.append("..")
logging.basicConfig(level=logging.INFO)

from src.trainer import Trainer
from src.model import BinaryCls
from src.data import RawDataProvider, IdiomDataset, collate_fn
from src.config import Config
from src.inference import Inference

from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

config = Config()
rdp = RawDataProvider("./data/train.csv", "./data/test.csv")

train, valid, _ = rdp.get_train_valid_test_samples(expected_extend_neg=2, max_neg_count=3)

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=config.max_length)
valid_set = IdiomDataset(valid, tokenizer=tokenizer, max_length=config.max_length)

train_dl = DataLoader(train_set, batch_size=config.batch_size, collate_fn=collate_fn)
valid_dl = DataLoader(valid_set, batch_size=config.batch_size, collate_fn=collate_fn)

bert_model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
model = BinaryCls(bert_model, bert_out_dim=config.bert_out_dim, hidden_dim=config.hidden_dim, cls_dim=1)

trainer = Trainer(train_dl, valid_dl, None, model, config)
inference = Inference(rdp, tokenizer, config.max_length, config, collate_fn, trainer)
trainer.load_inference(inference)

trainer.start_train()
