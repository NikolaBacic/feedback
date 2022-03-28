import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, Subset, DataLoader


class Collate:
    
    
    def __init__(self, tokenizer, purpose='train'):
        assert purpose in ['train', 'test'], 'purpose parameter should be train or test'
        self.tokenizer = tokenizer
        self.purpose = purpose
    
    
    def __call__(self, batch):
        
        output = dict()
        output['id'] = [sample['id'] for sample in batch]
        output['input_ids'] = [sample['input_ids'] for sample in batch]
        output['attention_mask'] = [sample['attention_mask'] for sample in batch]
        output['token_to_word'] = [sample['token_to_word'] for sample in batch]
        if self.purpose == 'train':
            output['target'] = [sample['target'] for sample in batch]
        
        batch_max = max([len(ids) for ids in output['input_ids']])
        
        output['input_ids'] = [sample + (batch_max-len(sample)) * [self.tokenizer.pad_token_id] for sample in output['input_ids']]
        output['attention_mask'] = [sample + (batch_max-len(sample)) * [0] for sample in output['attention_mask']]
        output['token_to_word'] = [sample + (batch_max-len(sample)) * [-1] for sample in output['token_to_word']]
        if self.purpose == 'train':
            output['target'] = [sample + (batch_max-len(sample)) * [-1] for sample in output['target']]

        output['input_ids'] = torch.tensor(output['input_ids'], dtype=torch.long)
        output['attention_mask'] = torch.tensor(output['attention_mask'], dtype=torch.long)
        output['token_to_word'] = torch.tensor(output['token_to_word'], dtype=torch.long)
        if self.purpose == 'train':
            output['target'] = torch.tensor(output['target'], dtype=torch.long)
    
        return output


class LongformerDataset(Dataset):
    """Dataset for the longformer model."""
    
    
    def __init__(self, data: pd.DataFrame, param, purpose='train'):
        self.data = data
        self.param = param
        assert purpose in ['train', 'test'], 'purpose parameter should be train or test'
        self.purpose = purpose
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        
        sample = dict()
        sample['id'] = self.data.loc[idx, 'id']
        sample['input_ids'] = self.data.loc[idx, 'input_ids']
        sample['attention_mask'] = self.data.loc[idx, 'attention_mask']
        sample['token_to_word'] = self.data.loc[idx, 'token_to_word']
        if self.purpose == 'train':
            sample['target'] = self.data.loc[idx, 'target']
            
        return sample
    
    
    def get_dataloaders(self, collate_fn, fold_idx):
        
        train_filter = (self.data['kfold'] != fold_idx)
        
        train_data = Subset(self, self.data.loc[train_filter].index)
        val_data = Subset(self, self.data.loc[~train_filter].index)

        train_dataloader = DataLoader(train_data,
                                      batch_size=self.param['batch_size'],
                                      collate_fn=collate_fn,
                                      shuffle=True,
                                      num_workers=0)

        val_dataloader = DataLoader(val_data,
                                    batch_size=self.param['batch_size'],
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=0)
        
        return train_dataloader, val_dataloader
        
        
    def get_test_dataloader(self, collate_fn):

        test_dataloader = DataLoader(self,
                                     batch_size=self.param['batch_size'],
                                     collate_fn=collate_fn,
                                     shuffle=False,
                                     num_workers=0)

        return test_dataloader 



