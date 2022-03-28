import os
import sys

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import transformers
from transformers import LongformerTokenizerFast

from tqdm import tqdm
import matplotlib.pyplot as plt

from param_longformer import param
from dataset_longformer import LongformerDataset, Collate
from model_longformer import init_longformer

# postprocess
from decode import decode_predictions

sys.path.append('/home/backe/projects/feedback/')
from utils import seed_everything, moving_average, score_feedback_comp

seed_everything(param['random_seed'])

os.environ['CUDA_VISIBLE_DEVICES'] = param['gpu_idx']
transformers.logging.set_verbosity_error()


def main():
    
    # LOAD PREPROCESSED DATA
    tokenizer = LongformerTokenizerFast.from_pretrained(param['model_name'])
    # load saved processed data
    DATA_PATH = '/storage/backe/feedback/data/longformer_preprocessed.csv'
    data = pd.read_csv(DATA_PATH)
    data['input_ids'] = data['input_ids'].apply(eval)
    data['attention_mask'] = data['attention_mask'].apply(eval)
    data['token_to_word'] = data['token_to_word'].apply(eval)
    data['target'] = data['target'].apply(eval)
    
    # GET DATALOADERS
    collate_fn = Collate(tokenizer, purpose='train')
    dataset = LongformerDataset(data, param, purpose='train')

    train_dataloader, val_dataloader = dataset.get_dataloaders(collate_fn, param['fold_idx'])
    
    # MODEL INITIALIZATION
    model = init_longformer(param)
    
    # VAL_DF
    TRAIN_PATH = '../data/train.csv'
    train_df = pd.read_csv(TRAIN_PATH)

    fold_ids = data.loc[data['kfold'] == param['fold_idx'], 'id']
    val_df = train_df[train_df['id'].isin(fold_ids)]
    
    # TRAIN - VALIDATE
    losses = model.train_eval_pipeline(train_dataloader, val_dataloader, val_df, score_feedback_comp, decode_predictions)
    
    # save model and tokenizer
    model.save_pretrained(param['save_dir'])
    tokenizer.save_pretrained(param['save_dir'])
    
if __name__ == '__main__':
    main()
    