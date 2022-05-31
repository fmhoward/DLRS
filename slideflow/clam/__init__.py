import os
import types

# internal imports
from slideflow.clam.utils.file_utils import save_pkl
from slideflow.clam.utils import *
from slideflow.clam.datasets.dataset_generic import Generic_MIL_Dataset
from slideflow.clam.utils.core_utils import train
from slideflow.clam.utils.eval_utils import *

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

def get_args(**kwargs):
    args_dict = {
        'num_splits': 1,
        'k': 3,
        'k_start': -1,
        'k_end': -1,
        'max_epochs': 20,
        'lr': 1e-4,
        'reg': 1e-5,
        'label_frac': 1,
        'bag_loss': 'ce',
        'bag_weight': 0.7,
        'model_type': 'clam_sb',
        'model_size': None,
        'use_drop_out': False,
        'drop_out': False,
        'weighted_sample': False,
        'opt': 'adam',
        'inst_loss': None,
        'no_inst_cluster': False,
        'B': 8,
        'log_data': False,
        'testing': False,
        'early_stopping': False,
        'subtyping': False,
        'seed': 1,
        'results_dir': None,
        'n_classes': None,
        'split_dir': None,
        'data_root_dir': None
    }
    for k in kwargs:
        args_dict[k] = kwargs[k]
    args = types.SimpleNamespace(**args_dict)
    return args

def detect_num_features(path_to_pt):
    features = torch.load(path_to_pt)
    return features.size()[1]

def main(args, dataset):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(ckpt_path, args, dataset):
    all_results, all_auc, all_acc = [], [], []
    model, patient_results, test_error, auc, df = eval(dataset, args, ckpt_path)
    all_results.append(all_results)
    all_auc.append(auc)
    all_acc.append(1-test_error)
    df.to_csv(os.path.join(args.save_dir, 'results.csv'), index=False)

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True