import numpy as np

discourse_map_reverse = {
    0: 'Filler',
    1: 'Claim',
    2: 'Claim',
    3: 'Evidence',
    4: 'Evidence',
    5: 'Position',
    6: 'Concluding Statement',
    7: 'Lead',
    8: 'Counterclaim',
    9: 'Rebuttal',
}

lead_info = {
    'p_start': 0.26,
    'p_end': 0.45,
    'min_conf': 0.53,
    'min_words': 5
}

pos_info = {
    'p_start': 0.51,
    'p_end': 0.46,
    'min_conf': 0.55,
    'min_words': 4
}

conc_info = {
    'min_word_prob': 0.37,
    'min_conf': 0.59,
    'min_words': 5
}

claim_info = {
    'min_conf': 0.53,
    'min_words': 2,
    'min_always': 15
}

evidence_info = {
    'min_conf': 0.62,
    'min_words': 16,
    'min_always': 40
}

count_info = {
    'min_conf': 0.45,
    'min_words': 4,
    'prob': 0.35
}

rebuttal_info = {
    'min_conf': 0.45,
    'min_words': 6,
    'prob': 0.31
}

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def find_max_list(lst):
    """ Return the index longest list of in the nested lists."""
    
    lst_tmp = [len(x) for x in lst]
    max_value = max(lst_tmp)
    max_index = lst_tmp.index(max_value)
    
    return max_index


def high_mean(arr):
    
    n = arr.shape[0]
    
    return arr[arr.argsort()[-(n//2):][::-1]].mean()


def decode_conc_stat(idx, word_probs, min_word_prob, min_conf, min_words, label_idx=6):
        
    sample_preds = []
    
    class_probs = word_probs[:, label_idx]
    
    candidates = np.where(class_probs >= min_word_prob)[0]
        
    if candidates.size > 1:
        
        candidates = group_consecutives(candidates)
        
        max_index = find_max_list(candidates)
        best_candidate = candidates[max_index]
        
        start_idx = best_candidate[0]
        end_idx = best_candidate[-1]
        
        num_word = end_idx - start_idx + 1
        confidence = high_mean(class_probs[start_idx:end_idx+1])

        if ((confidence>=min_conf) and (num_word>=min_words)) or (num_word > 100):
            # format prediction
            this_preds = [str(idx) for idx in range(start_idx, end_idx+1)]
            this_preds = ' '.join(this_preds)
            sample_preds.append([idx, discourse_map_reverse[label_idx], this_preds])

    return sample_preds


def decode_lead_position(idx, word_probs, p_start, p_end, min_conf, min_words, label_idx):
        
    sample_preds = []

    class_probs = word_probs[:, label_idx]
        
    # start index candidates
    start_candidates = np.where(class_probs >= p_start)[0]
    # end index candidates
    end_candidates = np.where(class_probs >= p_end)[0]
        
    if (len(start_candidates)>0) and (len(end_candidates)>0):
        start_idx = start_candidates[0]
        end_idx = end_candidates[-1]
        num_word = end_idx - start_idx + 1
        confidence = class_probs[start_idx:end_idx+1].mean()

        if (confidence>=min_conf) and (num_word>=min_words):
            # format prediction
            this_preds = [str(idx) for idx in range(start_idx, end_idx+1)]
            this_preds = ' '.join(this_preds)
            sample_preds.append([idx, discourse_map_reverse[label_idx], this_preds])

    return sample_preds


def get_claim_evidence_preds(idx, word_probs, start_label, body_label, min_conf, min_words, min_always):
    
    sample_preds = []
    
    start_probs = word_probs[:, start_label]
    body_probs = word_probs[:, body_label]

    word_preds = word_probs.argmax(1)
    
    num_words = len(word_preds)
    
    # clean word preds
    for i in range(1, num_words-1):
        if (word_preds[i-1] == 1 and  word_preds[i+1] == 1):
            if word_preds[i] == 3:
                word_preds[i] = 1
    # clean word preds
    for i in range(1, num_words-1):
        if (word_preds[i-1] == 3 and  word_preds[i+1] == 3):
            if word_preds[i] == 1:
                word_preds[i] = 3       

    start_positions = np.where(word_preds==start_label)[0]
    
    add_start = [i for i in range(1, num_words-1)
             if (word_preds[i] == body_label and word_preds[i-1] != body_label and word_preds[i-1] != start_label)]
    add_start = np.array(add_start)
    start_positions = np.append(start_positions, add_start).astype(int)

    for start_idx in start_positions:
        end_idx = start_idx
        
        while (end_idx != num_words-1) and (word_preds[end_idx+1] == body_label):
            end_idx += 1
        
        sample_num_word = end_idx - start_idx + 1
        confidence = np.append(start_probs[start_idx], body_probs[start_idx+1:end_idx+1]).mean()
        
        if ((confidence>=min_conf) and (sample_num_word>=min_words)) or (sample_num_word>=min_always):
            # format prediction
            this_preds = [str(idx) for idx in range(start_idx, end_idx+1)]
            this_preds = ' '.join(this_preds)
            sample_preds.append([idx, discourse_map_reverse[body_label], this_preds])
    
    return sample_preds


def get_count_reb_preds(idx, word_probs, label_idx, min_conf, min_words, prob):
        
    sample_preds = []
    
    class_probs = word_probs[:, label_idx]

    word_preds = np.where(class_probs > prob, 1, 0)
    
    num_words = len(word_preds)
    
    start_positions = [i for i in range(1, num_words-1)
             if (word_preds[i] == 1 and word_preds[i-1] != 1)]
    start_positions = np.array(start_positions)
    
    if word_preds[0] == 1:
        start_positions = np.append(0, start_positions).astype(int)
    
    for start_idx in start_positions:
        end_idx = start_idx
        
        while (end_idx != num_words-1) and (word_preds[end_idx+1] == 1):
            end_idx += 1
        
        sample_num_word = end_idx - start_idx + 1
        confidence = class_probs[start_idx:end_idx+1].mean()
        
        if (confidence>=min_conf) and (sample_num_word>=min_words):
            # format prediction
            this_preds = [str(idx) for idx in range(start_idx, end_idx+1)]
            this_preds = ' '.join(this_preds)
            sample_preds.append([idx, discourse_map_reverse[label_idx], this_preds])
    
    
    return sample_preds


def decode_predictions(idx, word_probs):
    
    preds_decoded = []
    
    # 1. Lead
    preds_decoded += decode_lead_position(idx, word_probs, lead_info['p_start'], lead_info['p_end'], lead_info['min_conf'], lead_info['min_words'], 7)
        
    # 2. Position
    preds_decoded += decode_lead_position(idx, word_probs, pos_info['p_start'], pos_info['p_end'], pos_info['min_conf'], pos_info['min_words'], 5)
        
    # 3. Concluding Statement
    preds_decoded += decode_conc_stat(idx, word_probs, conc_info['min_word_prob'], conc_info['min_conf'], conc_info['min_words'])
    
    # 4. Claim
    preds_decoded += get_claim_evidence_preds(idx, word_probs, 2, 1, claim_info['min_conf'], claim_info['min_words'], claim_info['min_always'])
    
    # 5. Evidence
    preds_decoded += get_claim_evidence_preds(idx, word_probs, 4, 3, evidence_info['min_conf'], evidence_info['min_words'], evidence_info['min_always'])
    
    # 6. Counterclaim
    preds_decoded += get_count_reb_preds(idx, word_probs, 8, count_info['min_conf'], count_info['min_words'], count_info['prob'])

    # 7. Rebuttal
    preds_decoded += get_count_reb_preds(idx, word_probs, 9, rebuttal_info['min_conf'], rebuttal_info['min_words'], rebuttal_info['prob'])

    
    return preds_decoded

