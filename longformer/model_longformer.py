import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import LongformerPreTrainedModel, LongformerModel, LongformerConfig
from transformers import AdamW, get_cosine_schedule_with_warmup

from torch.utils.checkpoint import checkpoint

from tqdm import tqdm

cel_loss = nn.CrossEntropyLoss()


class LongformerFeed(LongformerPreTrainedModel):
    
    def __init__(self, config, param):
        super().__init__(config)
        
        self.param=param
        
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.clf = nn.Linear(config.hidden_size, config.num_labels)
        
        nn.init.normal_(self.clf.weight, std=0.02)
        
        
    def forward(self, input_ids, attention_mask): 
        
        outputs = self.longformer(input_ids=input_ids,
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
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.param['warmup_steps'],
                                                    num_train_steps)

        return optimizer, scheduler
    
    
    def train_one_epoch(self, train_dataloader, optimizer, scheduler):

        losses = []

        self.train()

        for batch in tqdm(train_dataloader):

            # clean the gradients
            self.zero_grad()

            # add batch to GPU
            batch['input_ids'] = batch['input_ids'][:, :self.param['max_len']].cuda()
            batch['attention_mask'] = batch['attention_mask'][:, :self.param['max_len']].cuda()
            batch['token_to_word'] = batch['token_to_word'][:, :self.param['max_len']].cuda()
            batch['target'] = batch['target'][:, :self.param['max_len']].cuda()

            # get model's prediction
            output = self(batch['input_ids'],
                          attention_mask=batch['attention_mask'])    

            active_loss = batch['token_to_word'].view(-1) != -1
            active_logits = output.view(-1, self.config.num_labels)
            active_labels = torch.where(active_loss,
                                        batch['target'].view(-1),
                                        torch.tensor(cel_loss.ignore_index).type_as(batch['target']))

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

            # add batch to GPU
            batch['input_ids'] = batch['input_ids'].cuda()
            batch['attention_mask'] = batch['attention_mask'].cuda()

            with torch.no_grad():
                output = self(batch['input_ids'],
                              attention_mask=batch['attention_mask'])

            token_probs = torch.softmax(output, axis=-1).cpu()

            for i, sample_probs in enumerate(token_probs):
                token_to_word = batch['token_to_word'][i]
                num_words = token_to_word.max().item()+1

                word_probs = [sample_probs[token_to_word == word_id].mean(0).numpy() for word_id in range(num_words)]
                word_probs_all[batch['id'][i]] = np.array(word_probs)

        return word_probs_all
    
    
    def get_predictions(self, dataloader, decode_predictions):
        
        self.eval()

        # GET WORD LEVEL PROBABILITIES 
        word_probs_all = self.get_words_probabilities(dataloader)
        
        # PREDICTIONS FORMATTING
        val_preds = []

        for idx in word_probs_all.keys():

            word_probs = word_probs_all[idx]

            preds_decoded = decode_predictions(idx, word_probs)

            val_preds += preds_decoded

        preds_df = pd.DataFrame(val_preds, columns=['id', 'class', 'predictionstring'])
        preds_df.head()
        
        return preds_df
    
    
    def evaluate(self, dataloader, decode_predictions, val_df, score_feedback_comp):
        
        preds_df = self.get_predictions(dataloader, decode_predictions)
            
        score = score_feedback_comp(preds_df, val_df)
        print('Overall', score)
        print('#'*25 + '\n')
        
        return score
        
    
    def train_eval_pipeline(self, train_dataloader, val_dataloader, val_df, score_feedback_comp, decode_predictions):

        losses = []

        optimizer, scheduler = self.get_optimizer(train_dataloader)

        for epoch in range(self.param['epochs']-1):
            
            print(f'{epoch}. epoch training...')

            epoch_losses = self.train_one_epoch(train_dataloader, optimizer, scheduler)
            losses.append(epoch_losses)

            val_score = self.evaluate(val_dataloader, decode_predictions, val_df, score_feedback_comp)
            
        return losses, val_score

    
    
def init_longformer(param):
    
    config = LongformerConfig.from_pretrained(param['model_name'],
                                              num_labels=param['num_labels'],
                                              attention_window=param['attention_window'])
    model = LongformerFeed.from_pretrained(param['model_name'], config=config, param=param)
    model.cuda()
    model.gradient_checkpointing_enable()

    return model


def load_longformer(param):
    """use this when loading model for kaggle inference"""
    
    config = LongformerConfig.from_pretrained(param['kaggle_path'])
    model = LongformerFeed.from_pretrained(param['kaggle_path'], config=config, param=param)
    model.cuda()

    return model
