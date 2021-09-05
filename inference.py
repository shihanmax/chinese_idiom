import logging

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import IdiomDataset

logger = logging.getLogger(__name__)


class Inference(object):
    
    def __init__(
        self, raw_data_provider, tokenizer, max_length, config, collate_fn, 
        trainer,
    ):
        self.config = config
        self.raw_data_provider = raw_data_provider
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.collate_fn = collate_fn
        self.trainer = trainer
    
    def gen_submit_result(self, epoch):
        test_df = self.raw_data_provider.test_df
        length = len(test_df)
        logger.info(f"test length:{length}")
        
        all_samples = []
        for i in tqdm(range(length)):
            samples = self.raw_data_provider.build_data_samples_for_single(
                test_df.iloc[i], expected_extend_neg=-1, unique=False
            )
            all_samples.extend(samples)

        dataset = IdiomDataset(all_samples, self.tokenizer, self.max_length)

        data_loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            collate_fn=self.collate_fn
        )
        
        pred_res = []
        print("predicting...")
        self.trainer.model.eval()
        
        for data in tqdm(data_loader):
            probs, _, _ = self.trainer.forward_model(data)
            pred_res.append(probs.squeeze())

        self.trainer.model.train()
        
        all_pred = torch.cat(pred_res, dim=0).tolist()

        answer_idx = []
        for i in range(10):
            curr_prob = []
            for j in range(4):
                curr_prob.append(all_pred[i * 4 + j])
            answer_idx.append(curr_prob.index(max(curr_prob)))

        answers = []
        for i, idx in enumerate(answer_idx):
            answers.append(eval(test_df.iloc[i]["candidate"])[idx])

        df = pd.DataFrame({"label": answers})
        df.to_csv(f"./output/submit/submit-ep{epoch}.csv", index=False)
