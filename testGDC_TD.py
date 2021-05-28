print("start...")
import time
import yaml
import torch

from typing import List

import scipy.sparse as sp
import numpy as np
import seaborn as sns
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from data import get_dataset, HeatDataset, PPRDataset, set_train_val_test_split, get_adj_matrix
from models import GCN, GCNTD, GAT, JKNet, ARMA
from seeds import val_seeds, test_seeds

from scipy.linalg import expm

from torch_geometric.nn import GCNConv
from torch.nn import ModuleList, Dropout, ReLU, ELU
from TDConv import TDConv

class GDCTD(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 t: float,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GDCTD, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            # layers.append(SGConv(in_features, out_features, K=2))
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)
        self.diffusion = TDConv(num_features[0], t)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.diffusion.reset_parameters()


    def forward(self, data: Data, GDC_data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.diffusion(x, edge_index)
        for i, layer in enumerate(self.layers):
            x = layer(x, GDC_data.edge_index, edge_weight=GDC_data.edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)

with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)

device = 'cuda'

hidden_layers = 1
hidden_units = 64
lr = 0.01
weight_decay = 3
t_lr = 0.01
t = 3
num_per_class = 20
late_stop = True

dataset = get_dataset('Citeseer')
dataset2 = HeatDataset(
    name='Citeseer',
    use_lcc=True,
    t=4,
    k=None,
    eps=0.0009
)
dataset.data = dataset.data.to(device)
dataset2.data = dataset2.data.to(device)
# dataset2.data = dataset2.data.to(device)

model = GDCTD(
    dataset,
    t=t,
    hidden=hidden_layers * [hidden_units],
    dropout=0.5
).to(device)

#print(model)

def train(model: torch.nn.Module, optimizer: Optimizer, data: Data, key = "train"):
    model.train()
    optimizer.zero_grad()
    logits = model(data, dataset2.data)
    loss = F.nll_loss(logits[data[f'{key}_mask']], data.y[data[f'{key}_mask']])
    # for layer in model.layers:
    #     loss = loss - layer.t
    # print("loss: " + str(loss))
    loss.backward()
    optimizer.step()

def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data, dataset2.data)
    eval_dict = {}
    keys = ['val', 'test', 'train'] if test else ['val']
    for key in keys:
        mask = data[f'{key}_mask']
        # loss = F.nll_loss(logits[mask], data.y[mask]).item()
        # eval_dict[f'{key}_loss'] = loss
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        print(name)
        names = name.split('.')
        if len(set(names) & set(skip_list)) != 0:
            no_decay.append(param)
            # print("no_decay: " + name)
        else:
            decay.append(param)
            #print("decay: " + name)
    exit()
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def add_param(model, weight_decay, skip_list=[], contain_list=[]):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        #print(name)
        names = name.split('.')
        if len(set(names) & set(skip_list)) != 0:
            continue
            # print("no_decay: " + name)
        else:
            if len(contain_list) == 0:
                decay.append(param)
            else:
                if len(set(names) & set(contain_list)) != 0:
                    decay.append(param)
                else:
                    continue
            #print("decay: " + name)
    return [
        {'params': decay, 'weight_decay': weight_decay}]

def run(dataset: InMemoryDataset,
        model: torch.nn.Module,
        seeds: np.ndarray,
        test: bool = False,
        max_epochs: int = 10000,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        num_development: int = 1500,
        device: str = 'cuda'):
    start_time = time.perf_counter()

    best_dict = defaultdict(list)

    cnt = 0
    for seed in tqdm(seeds):
        dataset.data = set_train_val_test_split(
            seed,
            dataset.data,
            num_development=num_development,
            num_per_class=num_per_class
        ).to(device)
        model.to(device).reset_parameters()

        # skip_list = [str(i + 1) for i in range(hidden_layers)] + ["t"]
        # params = add_weight_decay(model, weight_decay, skip_list)
        # params_train_decay = add_param(model, weight_decay, skip_list=[str(i + 1) for i in range(hidden_layers)] + ["t"])
        # params_train_no_decay = add_param(model, 0, skip_list=["0"] + ["t"])
        params_train = [{'params': model.non_reg_params, 'weight_decay': 0.}, {'params': model.reg_params, 'weight_decay': weight_decay}]
        params_valid = add_param(model, 0, contain_list=["t"])


        optimizer = Adam(
            params_train,
            lr=lr
        )
        optimizer_val = Adam(
            params_valid,
            lr=t_lr
        )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        for epoch in range(1, max_epochs + 1):
            if patience_counter == patience:
                if late_stop == True:
                    if epoch > 300:
                        break
                    else:
                        patience_counter -= 1
                else:
                    break

            # if epoch == 100:
            #     model.layers[0].t.requires_grad = True
            train(model, optimizer, dataset.data, key = "train")
            # trainD(model, optimizer, dataset.data)
            train(model, optimizer_val, dataset.data, key = "val")
            eval_dict = evaluate(model, dataset.data, test)

            # if epoch % 10 == 0:
            #     print("epoch: " + str(epoch) + ", " + str(eval_dict))
            #     #print("t1: " + str(model.layers[0].t.data.cpu().numpy()) + "t2: " + str(model.layers[1].t.data.cpu().numpy()))
            #     print("t: " + str(model.diffusion.t.data.cpu().numpy()))
            
            if eval_dict['val_acc'] <= tmp_dict['val_acc']:
                patience_counter += 1
            else:
                patience_counter = 0
                tmp_dict['epoch'] = epoch
                for k, v in eval_dict.items():
                    tmp_dict[k] = v
        # for layer in model.layers:
        #     print(layer.t)
        cur_dict = {}
        for k, v in tmp_dict.items():
            best_dict[k].append(v)
            cur_dict[k] = v
        print(cur_dict)
        

    best_dict['duration'] = time.perf_counter() - start_time
    return dict(best_dict)

results = run(
    dataset,
    model,
    seeds=test_seeds if config['test'] else val_seeds,
    lr=lr,
    weight_decay=weight_decay,
    test=config['test'],
    num_development=config['num_development'],
    device=device
)

# print(results)

boots_series = sns.algorithms.bootstrap(results['val_acc'], func=np.mean, n_boot=1000)
results['val_acc_ci'] = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(results['val_acc'])))
if 'test_acc' in results:
    boots_series = sns.algorithms.bootstrap(results['test_acc'], func=np.mean, n_boot=1000)
    results['test_acc_ci'] = np.max(
        np.abs(sns.utils.ci(boots_series, 95) - np.mean(results['test_acc']))
    )

for k, v in results.items():
    if 'acc_ci' not in k and k != 'duration':
        results[k] = np.mean(results[k])


mean_acc = results['test_acc']
uncertainty = results['test_acc_ci']
print(f"Mean accuracy: {100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%")