import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup

from torch.utils.checkpoint import checkpoint

from tqdm import tqdm

cel_loss = nn.CrossEntropyLoss()


class RobertaFeed(RobertaPreTrainedModel):
    
    def __init__(self, config, param):
        super().__init__(config)
        
        self.param = param
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.clf = nn.Linear(config.hidden_size, config.num_labels)
        
        nn.init.normal_(self.clf.weight, std=0.02)
        
        
    def forward(self, input_ids, attention_mask): 
        
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)
        
        outputs = outputs[0]
        
        outputs = self.dropout(outputs)
        outputs = self.clf(outputs)
        
        return outputs
    
    
    def get_optimizer(self, train_dataloader):

        # define optimizer - Adam
        optimizer = AdamW(self.parameters(), lr=self.param['lr'])
        # define lr scheduler
        num_train_steps = self.param['epochs'] * len(train_dataloader)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              self.param['warmup_steps'],
                                                              num_train_steps,
                                                              lr_end=self.param['lr_end'],
                                                              power=self.param['power'])

        return optimizer, scheduler

    
    def train_one_epoch(self, train_dataloader, optimizer, scheduler):

        losses = []

        self.train()

        for batch in tqdm(train_dataloader):
            
            # clean the gradients
            self.zero_grad()

            # add batch to GPU
            batch = tuple(t.cuda() for t in batch)

            # unpack
            input_ids, attention_mask, loss_mask, target = batch

            # get model's prediction
            output = self(input_ids, attention_mask)    

            active_loss = loss_mask.view(-1) == 1
            active_logits = output.view(-1, self.config.num_labels)
            active_labels = torch.where(active_loss,
                                        target.view(-1),
                                        torch.tensor(cel_loss.ignore_index).type_as(target))

            loss = cel_loss(active_logits, active_labels)
            losses.append(loss.item())

            # calculate gradients
            loss.backward()

            # update model param
            optimizer.step()

            # update the learning rate
            scheduler.step()
            
        return losses

    
    def get_words_probabilities(self, dataloader):
        
        self.eval()

        word_probs_all = dict()

        for batch in dataloader:

            with torch.no_grad():
                output = self(batch['input_ids'].cuda(),
                              attention_mask=batch['attention_mask'].cuda())

            sample_probs = torch.softmax(output, axis=-1).cpu()
            sample_probs = sample_probs.view(-1, self.param['num_labels'])

            loss_mask = batch['loss_mask'].view(-1) == 1

            token_probs = sample_probs[loss_mask]

            token_to_word = batch['token_to_word'].reshape(-1)
            token_to_word = token_to_word[loss_mask]

            num_words = token_to_word.max().item()+1

            word_probs = [token_probs[token_to_word == word_id].mean(0).numpy() for word_id in range(num_words)]
            word_probs_all[batch['id']] = np.array(word_probs)

        return word_probs_all
    
    
    def get_predictions(self, dataloader, decode_predictions, model_decoder):
        
        self.eval()

        # GET WORD LEVEL PROBABILITIES 
        word_probs_all = self.get_words_probabilities(dataloader)
        
        preds_df = decode_predictions(model_decoder, word_probs_all)
            
        return preds_df
    
    
    def evaluate(self, dataloader, decode_predictions, model_decoder, val_df, score_feedback_comp):
        
        preds_df = self.get_predictions(dataloader, decode_predictions, model_decoder)
            
        score = score_feedback_comp(preds_df, val_df)
        print('Overall', score)
        print('#'*25 + '\n')
        
        return score
        
    
    def train_eval_pipeline(self, train_dataloader, val_dataloader, val_df, score_feedback_comp, decode_predictions, model_decoder):

        losses = []

        optimizer, scheduler = self.get_optimizer(train_dataloader)

        for epoch in range(self.param['epochs']):
            
            print(f'{epoch}. epoch training...')

            epoch_losses = self.train_one_epoch(train_dataloader, optimizer, scheduler)
            losses.append(epoch_losses)

            val_score = self.evaluate(val_dataloader, decode_predictions, model_decoder, val_df, score_feedback_comp)
            
        return losses, val_score

    
def init_roberta(param):
    
    config = RobertaConfig.from_pretrained(param['model_name'],
                                           num_labels=param['num_labels'],
                                           classifier_dropout=0.1)
    
    model = RobertaFeed.from_pretrained(param['model_name'], config=config, param=param)
    model.cuda()
    model.gradient_checkpointing_enable()

    return model


def load_roberta(param):
    """use this when loading model for kaggle inference"""
    
    config = RobertaConfig.from_pretrained(param['kaggle_path'])
    model = RobertaFeed.from_pretrained(param['kaggle_path'], config=config, param=param)
    model.cuda()
    
    return model
