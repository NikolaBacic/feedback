import os
import sys

import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn

import transformers
from transformers import RobertaTokenizerFast

from tqdm import tqdm
import matplotlib.pyplot as plt

from param_roberta import param
from dataset_roberta import RobertaDataset
from model_roberta import init_roberta

sys.path.append('/home/backe/projects/feedback/')
from utils import seed_everything, score_feedback_comp
from decoder_2000.candidates import decode_predictions

seed_everything(param['random_seed'])

os.environ['CUDA_VISIBLE_DEVICES'] = param['gpu_idx']
transformers.logging.set_verbosity_error()


def main():
    
    # LOAD PREPROCESSED DATA
    tokenizer = RobertaTokenizerFast.from_pretrained(param['model_name'])
    # load saved processed data
    DATA_PATH = '/DATA/backe/feedback/data/roberta_preprocessed.csv'
    data = pd.read_csv(DATA_PATH)
    data['input_ids'] = data['input_ids'].apply(eval)
    data['attention_mask'] = data['attention_mask'].apply(eval)
    data['token_to_word'] = data['token_to_word'].apply(eval)
    data['target'] = data['target'].apply(eval)
    
    # GET DATALOADERS
    dataset = RobertaDataset(data, tokenizer, param)

    train_dataloader, val_dataloader = dataset.get_dataloaders(param['fold_idx'])
    
    # MODEL INITIALIZATION
    model = init_roberta(param)
    
    # VAL_DF
    TRAIN_PATH = '../data/train.csv'
    train_df = pd.read_csv(TRAIN_PATH)

    fold_ids = data.loc[data['kfold'] == param['fold_idx'], 'id']
    val_df = train_df[train_df['id'].isin(fold_ids)]
    
    # lgbm decoder model
    with open('../decoder_2000/decoder_model.pickle', 'rb') as handle:
        model_decoder = pickle.load(handle)

    # TRAIN - VALIDATE
    losses = model.train_eval_pipeline(train_dataloader, val_dataloader, val_df, score_feedback_comp, decode_predictions, model_decoder)
    
    # save model and tokenizer
    model.save_pretrained(param['save_dir'])
#     tokenizer.save_pretrained(param['save_dir'])
    
if __name__ == '__main__':
    main()
