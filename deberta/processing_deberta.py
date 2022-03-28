import numpy as np
import pandas as pd

from tqdm import tqdm


discourse_map = {
    'Filler': 0,
    'Claim': 1,
    'Claim_S': 2,
    'Evidence': 3,
    'Evidence_S': 4,
    'Position': 5,
    'Concluding Statement': 6,
    'Lead': 7,
    'Counterclaim': 8,
    'Rebuttal': 9,
}

def clean_offset(pos: tuple, text:str) -> tuple:
    
    if pos[0] == pos[1]:
        return pos
    elif text[pos[0]] == ' ':
        new_start = pos[0] + 1
        return (new_start, pos[1])
    else:
        return pos


def preprocess(text_data, tokenizer, train_df=None):
                
    data_pp = []    

    for idx, text in tqdm(text_data.items()):
        
        # right strip the text
        text = text.rstrip()
        
        # 1. GET INPUTS
        inputs = tokenizer(text,
                           add_special_tokens=True,
                           return_offsets_mapping=True,
                           return_length=True)
        
        # clean first whitespace position
        inputs['offset_mapping'] = [clean_offset(pos, text) for pos in inputs['offset_mapping']]
        
        # 2. CREATE TOKEN TO WORD MAPPING
        # Note: -1 index designates tokens that don't belong to any word
        
        # split text into words
        words = text.split()

        token_to_word = [] # list to store token -> word mapping
        word_pos = 0 # starting word position

        tokens = inputs['input_ids'][1:-1]  # exclude <s> and </s> tokens
        start = 0
        end = 1

        for _ in tokens:

            word = tokenizer.decode(tokens[start:end]).strip()
            
            # if striped word is an empty string, that token doesn't belong to any word
            if word == '':
                token_to_word.append(-1)
                start += 1
                end += 1
                continue
                
            # still no match
            # continue adding tokens
            if word != words[word_pos]:
                end += 1
                token_to_word.append(word_pos)
            # match 
            else:
                token_to_word.append(word_pos)
                start = end
                end = start + 1
                word_pos += 1
        
        # add -1 position for the <s> and </s> tokens        
        token_to_word = [-1] + token_to_word + [-1]
        
        
        # 3. FORM TARGET VECTOR     
        if train_df is not None:
            
            # initialize target 0s (all Fillers)
            target = np.full(inputs['length'][0], 0)
            id_filt = (train_df['id'] == idx)
            sample_df = train_df[id_filt]
            
            # helper numpy array
            token_to_word_np = np.array(token_to_word)
            
            # iterate discourses
            for row in sample_df.iterrows():
                discourse_type = row[1]['discourse_type']
                start = row[1]['new_start']
                end = row[1]['new_end']
                
                # this discourse's token positions
                # set their targets
                discourse_pos = [True if ((pos[0] >= start) and (pos[1] <= end)) else False for pos in inputs['offset_mapping']]
                target[discourse_pos] = discourse_map[discourse_type]

                # special first word's token's target for Claim and Evidence
                # set their target to Claim_S / Evidence_S 
                if (discourse_type == 'Claim') or (discourse_type == 'Evidence'):
                    first_word_id = int(row[1]['new_predictionstring'].split()[0])
                    target[token_to_word_np == first_word_id] = discourse_map[discourse_type + '_S']

            # tokens that doesn't belong to any word set to -1
            # easier this way at the end...
            target[token_to_word_np == -1] = -1
            target = list(target)
                
        if train_df is not None:
            data_pp.append([idx, inputs['input_ids'], inputs['attention_mask'], token_to_word,  target])
        else:
            data_pp.append([idx, inputs['input_ids'], inputs['attention_mask'], token_to_word])
    
    
    return data_pp
    