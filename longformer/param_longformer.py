param = {
    
    'model_name': 'allenai/longformer-large-4096',
    'save_dir': './longformer-1fold/',
    'kaggle_path': '../input/longformer-1fold/',
    
    'random_seed': 39,
    'gpu_idx': '0',
    
    'fold_idx': 1,
    
    'max_len': 1024,
    'num_labels': 10,
    
    'attention_window': 512,
    
    'batch_size': 4,
    'epochs': 4,
    'lr': 0.0000125,
    'warmup_steps': 350
    
}
