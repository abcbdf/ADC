import argparse
import torch

def get_citation_args():
    if not hasattr(get_citation_args, "args"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--t', type=float, default=3)
        parser.add_argument('--latestop', action='store_true', default=False)
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--preprocessing', type=str, default='none')
        parser.add_argument('--fixT', action='store_true', default=False)
        parser.add_argument('--debugInfo', action='store_true', default=False)
        parser.add_argument('--step', type=int, default=10)
        parser.add_argument('--denseT', action='store_true', default=False)
        parser.add_argument('--shareT', action='store_true', default=False)
        parser.add_argument('--lateDiffu', action='store_true', default=False)
        parser.add_argument('--swapTrainValid', action='store_true', default=False)
        parser.add_argument('--tLr', type=float, default=0.01)
        parser.add_argument('--num_per_class', type=int, default=20)
        
        get_citation_args.args = parser.parse_args()

        if get_citation_args.args.shareT == True:
            assert(get_citation_args.args.denseT == True)
    return get_citation_args.args