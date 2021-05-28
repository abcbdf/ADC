print("start...")
import time
import yaml
import torch

import scipy.sparse as sp
import numpy as np
import seaborn as sns
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from data import get_dataset, HeatDataset, PPRDataset, get_adj_matrix
from ImpModels import GCN, JKNet, ARMA
from seeds import val_seeds, test_seeds

from scipy.linalg import expm
from args import get_citation_args
from seeds import development_seed

args = get_citation_args()

with open("./config/" + args.config, 'r') as c:
    config = yaml.safe_load(c)

device = 'cuda'

preprocessing = args.preprocessing

# hidden_layers = 1
# hidden_units = 16
# lr = 0.01
# weight_decay = 0.00
# t_lr = 0.01
# t = 3
# num_per_class = 20
# late_stop = False

dataset = get_dataset(config['dataset_name'])

dataset.data = dataset.data.to(device)

model_parameter = {
    'dataset': dataset, 
    'hidden': config[preprocessing]['hidden_layers'] * [config[preprocessing]['hidden_units']],
    'dropout': config[preprocessing]['dropout']
}
model_parameter['t'] = args.t
if config['architecture'] == 'ARMA':
    model_parameter['stacks'] = config[preprocessing]['stacks']

model = globals()[config['architecture']](**model_parameter).to(device)
assert(not hasattr(model, "diffusion"))
#print(model)

def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    t_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        cur_train_idx = rnd_state.choice(class_idx, num_per_class, replace=False)
        new_class_idx = [i for i in class_idx if i not in cur_train_idx]
        cur_t_idx = rnd_state.choice(new_class_idx, num_per_class, replace=False)
        train_idx.extend(cur_train_idx)
        t_idx.extend(cur_t_idx)
        
    merge_idx = train_idx + t_idx
    val_idx = [i for i in development_idx if i not in merge_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.t_mask = get_mask(t_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

def train(model: torch.nn.Module, optimizer: Optimizer, data: Data, key = "train"):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data[f'{key}_mask']], data.y[data[f'{key}_mask']])
    # for layer in model.layers:
    #     loss = loss - layer.t
    # print("loss: " + str(loss))
    loss.backward()
    optimizer.step()

def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    eval_dict = {}
    keys = ['val', 'test', 'train', "t"] if test else ['val']
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
        #print(name)
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
            num_per_class=config["num_per_class"]
        ).to(device)
        if args.swapTrainValid == True:
            dataset.data.train_mask, dataset.data.val_mask = dataset.data.val_mask, dataset.data.train_mask
            #dataset.data.val_mask = dataset.data.train_mask + dataset.data.val_mask
        model.to(device).reset_parameters()

        # skip_list = [str(i + 1) for i in range(hidden_layers)] + ["t"]
        # params = add_weight_decay(model, weight_decay, skip_list)
        params_train_decay = add_param(model.layers[0], weight_decay, skip_list=["t"])
        params_train_no_decay = []
        for layer in model.layers[1:]:
            params_train_no_decay += add_param(layer, 0, skip_list=["t"])
        #params_train_decay = add_param(model, weight_decay, skip_list=[str(i + 1) for i in range(hidden_layers)] + ["t"])
        #params_train_no_decay = add_param(model, 0, skip_list=["0"] + ["t"])
        # params_train = [{'params': model.non_reg_params, 'weight_decay': 0.}, {'params': model.reg_params, 'weight_decay': weight_decay}]
        params_train = params_train_decay + params_train_no_decay
        params_valid = add_param(model, 0, contain_list=["t"])
        

        optimizer = Adam(
            params_train,
            lr=lr
        )
        optimizer_val = Adam(
            params_valid,
            lr=args.tLr
        )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        for epoch in range(1, max_epochs + 1):
            if patience_counter == patience:
                if args.latestop == True:
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
            if not args.fixT:
                train(model, optimizer_val, dataset.data, key = "t")
            eval_dict = evaluate(model, dataset.data, test)

            if epoch % 10 == 0 and args.debugInfo:
                print("epoch: " + str(epoch) + ", " + str(eval_dict))
                print("t1: " + str(model.layers[0].diffusion.t.data.cpu().numpy()) + "t2: " + str(model.layers[1].diffusion.t.data.cpu().numpy()))
            #     print("t1: " + str(model.layers[0].t.data.cpu().numpy()) + "t2: " + str(model.layers[1].t.data.cpu().numpy()))
            #     #print("t: " + str(model.diffusion.t.data.cpu().numpy()))
            
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
    lr=config[preprocessing]['lr'],
    weight_decay=config[preprocessing]['weight_decay'],
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