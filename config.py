import os
import torch


class Config(object):
    max_length = 512
    
    epoch = 10
    num_warmup_epochs = 3
    not_early_stopping_at_first = 5
    es_with_no_improvement_after = 5
    start_gen_submit = 3
    batch_size = 16
    
    verbose = 3
    
    lr = 3e-5
    gradient_clip = 5
    
    bert_out_dim = 768
    hidden_dim = 512
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "./output/summary/")
    model_path = os.path.join(base_dir, "./output/model/finetuned")
    log_path = os.path.join(base_dir, "./output/log/train_log.log")
