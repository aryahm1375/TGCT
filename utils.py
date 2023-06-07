import numpy as np

import torch
import json
import random
import copy
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    average_precision_score, 
    auc,
    roc_curve,
    precision_recall_curve)
from sklearn.metrics import precision_recall_fscore_support
class eICUDataset(Dataset):
    def __init__(self, data):
        print('LEN_DATA',len(data))
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def priors_collate_fn(batch):
    new_batch = []
    for i, item in enumerate(batch):
        num_indices = item[0].shape[-1]
        new_indices = torch.cat((torch.tensor([i]*num_indices).reshape(1,-1), item[0]), axis=0)
        new_batch.append((new_indices, item[1]))
    indices = torch.cat([t[0] for t in new_batch], axis=1)
    values = torch.cat([t[1] for t in new_batch], axis=-1)
    return indices, values

    
    
def get_extended_attention_mask(attention_mask):
    if attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps-num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def nested_concat(tensors, new_tensors, dim=0):
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def nested_numpify(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def prepare_data(data, priors_data, device, num):
    features = {}
    features['dx_ints'] = data[0]
    features['proc_ints'] = data[1]
    features['dx_masks'] = data[2]
    features['proc_masks'] = data[3]

    #features['readmission'] = data[4]
    features['expired'] = data[4]
    features['patient_id'] = data[5]
    #print('index',ind,data[4])
    #print('EXPITREDRES',data[4])
    for k, v in features.items():
        features[k] = v.to(device)
    priors = {}
    priors['indices'] = priors_data[0].to(device)
    priors['values'] = priors_data[1].to(device)
        
    
    return features, priors
    
def compute_metrics(preds1, labels, description = None):
    metrics = {}
    preds1 = preds1.squeeze(1).cpu().numpy()
    asgar = copy.copy(preds1)
    asgar[preds1>0.75] = 1
    asgar[preds1 <= 0.75] = 0
    preds = np.array(preds1)

    precisions, recalls, thresholds = precision_recall_curve(labels, preds)

    auc_pr = auc(recalls, precisions)
    metrics['AUCPR'] = auc_pr
    if description == 'Evaluating':
        plt.plot(recalls, precisions)
        plt.savefig('eval.png')

    #fpr, tpr, thresholds = roc_curve(labels, preds)
    auc_roc = roc_auc_score(labels, preds)
    metrics['AUROC'] = auc_roc
    p,r,f,_ = precision_recall_fscore_support(labels, asgar, average='binary')
    metrics['Precision'] = p
    metrics['Recall'] = r
    metrics['F1'] = f
    metrics['Accuracy'] = accuracy_score(labels, asgar)
    p, r, f, _ = precision_recall_fscore_support(labels, asgar, average='weighted')
    metrics['Precision_w'] = p
    metrics['Recall_w'] = r
    metrics['F1_w'] = f

    
    
    return metrics

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--max_num_codes', type=int, default=50)
        self.add_argument('--feature_keys', action='append', default=['dx_ints','proc_ints'])
        self.add_argument('--vocab_sizes', type=json.loads, default={'dx_ints1':3249, 'proc_ints1':2210,'dx_ints2':3249, 'proc_ints2':2210,'dx_ints3':3249, 'proc_ints3':2210})
        self.add_argument('--prior_scalar', type=float, default=0.5)
        
        self.add_argument('--num_stacks', type=int, default=6)#stack = 3
        self.add_argument('--hidden_size', type=int, default=128)#128
        self.add_argument('--intermediate_size', type=int, default=512)#256
        self.add_argument('--num_heads', type=int, default=1)
        self.add_argument('--hidden_dropout_prob', type=float, default=0.002)

        
        self.add_argument('--learning_rate', type=float, default=1e-2)
        self.add_argument('--eps', type=float, default=1e-8)
        self.add_argument('--batch_size', type=int, default=256)
        self.add_argument('--max_grad_norm', type=float, default=1.0)
        
        self.add_argument('--use_guide', default=False, action='store_true')
        self.add_argument('--use_prior', default=False, action='store_true')

        self.add_argument('--output_hidden_states', default=False, action='store_true')
        self.add_argument('--output_attentions', default=False, action='store_true')
        
        self.add_argument('--fold', type=int, default=0)
        self.add_argument('--eval_batch_size', type=int, default=512)
        
        self.add_argument('--warmup', type=float, default=0.05)
        self.add_argument('--logging_steps', type=int, default=100)
        self.add_argument('--max_steps', type=int, default=100000)
        self.add_argument('--num_train_epochs', type=int, default=0)
        self.add_argument('--data_dir', type=str, default = 'Processed_raw/')
        self.add_argument('--output_dir', type=str, default='expired')

        self.add_argument('--label_key', type=str, default='expired')
        self.add_argument('--num_labels', type=int, default=2)
        
        self.add_argument('--reg_coef', type=float, default=0)
        self.add_argument('--seed', type=int, default=42)
        self.add_argument('--cuda', type=int, default=1)
        
        self.add_argument('--do_train', default=True, action='store_true')
        self.add_argument('--do_eval', default=False, action='store_true')
        self.add_argument('--do_test', default=True, action='store_true')

    def parse_args(self):
        args = super().parse_args()
        return args
