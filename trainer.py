import logging

import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import accuracy_score

from nlkit.trainer import BaseTrainer
from nlkit.utils import (
    Phase, get_linear_schedule_with_warmup_ep, weight_init,
    check_should_do_early_stopping,
)

logger = logging.getLogger('__name__')


class Trainer(BaseTrainer):
    
    def __init__(self, train_dl, valid_dl, test_dl, model, config):
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=self.config.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup_ep(
            self.optimizer, 
            num_warmup_epochs=self.config.num_warmup_epochs,
            total_epochs=self.config.epoch
        )
        
        super(Trainer, self).__init__(
            model, train_dl, valid_dl, test_dl, self.lr_scheduler, 
            self.optimizer, weight_init, self.config.summary_path, 
            self.config.device, None, self.config.epoch, 
            self.config.model_path,
        )
        
        self.loss_record_on_valid = []
        self.train_record = []
        self.log_file = open(self.config.log_path, "a+")
        
    def load_inference(self, inference):
        self.inference = inference
        
    def test(self, epoch):
        if epoch >= self.config.start_gen_submit:
            self.inference.gen_submit_result(epoch)
    
    def handle_summary(self, phase, log_info):
        
        if phase is Phase.TRAIN:
            global_step = self.global_train_step
        elif phase is Phase.VALID:
            global_step = self.global_valid_step
        elif phase is Phase.TEST:
            global_step = self.global_test_step
        else:
            raise ValueError(f"invalid phase:{phase.name}")
        
        for k, v in log_info.items():
            if k not in {"curr_loss", "avg_loss"}:
                continue
            
            self.summary_writer.add_scalar(
                tag=f"{phase.name}/{k}", 
                scalar_value=v, 
                global_step=global_step,
            )
    
    def start_train(self):
        try:
            super().start_train()
        except KeyboardInterrupt:
            self.log_file.close()
        finally:
            self.log_file.close()
        
    def iteration(self, epoch, data_loader, phase):
        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP:{}:{}".format(phase.name, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        total_loss = []

        for idx, data in data_iter:
            
            if phase == Phase.TRAIN:
                self.global_train_step += 1
            elif phase == Phase.VALID:
                self.global_valid_step += 1
            else:
                self.global_test_step += 1

            # data to device
            data = {key: value.to(self.device) for key, value in data.items()}

            # forward the model
            if phase == Phase.TRAIN:
                _, loss, pred = self.forward_model(data)
                
            else:
                with torch.no_grad():
                    _, loss, pred = self.forward_model(data)
                    
            total_loss.append(loss.item())

            # do backward if on train
            if phase == Phase.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()

                if self.config.gradient_clip:
                    clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip,
                    )
                    
                self.optimizer.step()

            curr_acc = accuracy_score(data["label"].cpu(), pred.cpu())
            
            log_info = {
                "phase": phase.name,
                "epoch": epoch,
                "iter": idx,
                "curr_loss": loss.item(),
                "avg_loss": sum(total_loss) / len(total_loss),
                "acc": curr_acc,
            }
            
            self.handle_summary(phase, log_info)
            
            if self.config.verbose and not idx % self.config.verbose:
                data_iter.write(str(log_info))
                self.log_file.write(f"{log_info}\n")
                self.log_file.flush()

        if phase == Phase.TRAIN:
            self.lr_scheduler.step()  # step every train epoch

        avg_loss = sum(total_loss) / len(total_loss)
        
        logger.info(
            "EP:{}_{}, avg_loss={}".format(
                epoch,
                phase.name,
                avg_loss,
            ),
        )

        # 记录训练信息
        record = {
            "epoch": epoch,
            "status": phase.name,
            "avg_loss": avg_loss,
        }

        self.train_record.append(record)

        # check should early stopping at valid
        if phase == Phase.VALID:
            self.loss_record_on_valid.append(avg_loss)

            should_stop = check_should_do_early_stopping(
                self.loss_record_on_valid,
                self.config.not_early_stopping_at_first,
                self.config.es_with_no_improvement_after,
                acc_like=False,
            )

            if should_stop:
                best_epoch = should_stop
                logger.info("Now stop training..")
                return best_epoch
        
        return False

    def forward_model(self, data, with_loss=True):
        probs = self.model(
            input_ids=data["input_ids"],
            token_type_ids=data["token_type_ids"], 
            attention_mask=data["attention_mask"],
            position_ids=data["position_ids"],
        )
        pred = probs.argmax(dim=-1)
        
        if with_loss:
            loss = self.model.forward_loss(probs, data["label"])
        else:
            loss = None
        
        return probs, loss, pred
