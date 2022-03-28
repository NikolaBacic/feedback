import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from typing import List


class DebertaDataset:
    
    size = 512
    step = 128
    
    def __init__(self, data: pd.DataFrame, tokenizer, param):
        self.data = data
        self.tokenizer = tokenizer
        self.param = param
    
    def _preprocess_sample(self, sample: pd.Series, purpose: str):
        
        assert purpose in ['train', 'val', 'test'], 'purpose parameter should be train, val or test.'

        sample_len = len(sample['input_ids'])

        if sample_len <= self.size:
            
            pad_len = self.size-sample_len
            
            input_ids = torch.tensor(sample['input_ids'] + pad_len*[self.tokenizer.pad_token_id], dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor(sample['attention_mask'] + pad_len*[0], dtype=torch.long).unsqueeze(0)
            token_to_word = torch.tensor(sample['token_to_word'] + pad_len*[-1], dtype=torch.long).unsqueeze(0)
            
            loss_mask = torch.clone(attention_mask)
            loss_mask[token_to_word == -1] = 0
            
            if purpose != 'test':
                target = torch.tensor(sample['target'] + pad_len*[-1], dtype=torch.long).unsqueeze(0)                

        else:
            sample_len = len(sample['input_ids'])
            rest = sample_len%self.step
            pad_len = self.step-rest

            input_ids = torch.tensor(sample['input_ids'] + pad_len*[self.tokenizer.pad_token_id], dtype=torch.long)
            input_ids = input_ids.unfold(0, self.size, self.step)

            attention_mask = torch.tensor(sample['attention_mask'] + pad_len*[0], dtype=torch.long)
            attention_mask = attention_mask.unfold(0, self.size, self.step)
            
            token_to_word = torch.tensor(sample['token_to_word'] + pad_len*[-1], dtype=torch.long)
            token_to_word = token_to_word.unfold(0, self.size, self.step)

            loss_mask = torch.zeros(attention_mask.shape)
            n_sample = loss_mask.size(0)

            loss_mask[0, :2*self.step] = 1

            if n_sample==2:
                loss_mask[-1, self.step:3*self.step+rest] = 1
            if n_sample>=3:
                loss_mask[1, self.step:3*self.step] = 1
                loss_mask[2:-1, 2*self.step:3*self.step] = 1 # nothing when n_sample=3
                loss_mask[-1, 2*self.step:3*self.step+rest] = 1   
            
            loss_mask[token_to_word == -1] = 0
            
            if purpose != 'test':
                target = torch.tensor(sample['target'] + pad_len*[-1], dtype=torch.long)
                target = target.unfold(0, self.size, self.step)

        output = dict()
        output['input_ids'] = input_ids
        output['attention_mask'] = attention_mask
        output['loss_mask'] = loss_mask
        if purpose != 'test':
            output['target'] = target
        if purpose in ['val', 'test']:
            output['id'] = sample['id']
            output['token_to_word'] = token_to_word
        
        return output
    
    
    def _process_train_data(self, train_df:pd.DataFrame) -> TensorDataset:
        
        input_ids_t = []
        attention_mask_t = []
        loss_mask_t = []
        target_t = []
        
        for sample in train_df.iterrows():
            output = self._preprocess_sample(sample[1], 'train')

            input_ids_t.append(output['input_ids'])
            attention_mask_t.append(output['attention_mask'])
            loss_mask_t.append(output['loss_mask'])
            target_t.append(output['target'])
        
        train_data = TensorDataset(torch.cat(input_ids_t),
                                   torch.cat(attention_mask_t),
                                   torch.cat(loss_mask_t),
                                   torch.cat(target_t))
        return train_data

    
    def _process_val_data(self, df:pd.DataFrame, purpose) -> List[dict]:
        
        val_dataloader = []
        
        for sample in df.iterrows():
            output = self._preprocess_sample(sample[1], purpose)
            val_dataloader.append(output)

        
        return val_dataloader
    
    def get_dataloaders(self, fold_idx):
        
        # split the data
        train_filter = (self.data['kfold'] != fold_idx)
        train_df = self.data.loc[train_filter, ['input_ids', 'attention_mask', 'token_to_word', 'target']]
        val_df = self.data.loc[~train_filter, ['id', 'input_ids', 'attention_mask', 'token_to_word', 'target']]
        
        # train dataloader
        train_data = self._process_train_data(train_df)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.param['batch_size'])
        
        # validation dataloader
        val_dataloader = self._process_val_data(val_df, 'val')
        
        return train_dataloader, val_dataloader
    
    
    def get_test_dataloader(self):
        
        test_dataloader = self._process_val_data(self.data, 'test')
        
        return test_dataloader
        
