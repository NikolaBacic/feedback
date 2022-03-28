param = {
    
    'model_name': 'microsoft/deberta-large',
    'save_dir': './deberta-3fold/',
    'kaggle_path': '../input/deberta-3fold/',
    
    'random_seed': 23,
    'gpu_idx': '0',
    
    'fold_idx': 3,
    
    'num_labels': 10,
    
    'batch_size': 8,
    'epochs': 3,
    'lr': 1.75e-05,
    'lr_end': 1e-06,
    'power': 2.0,
    'warmup_steps': 400
    
}
