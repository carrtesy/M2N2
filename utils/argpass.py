import argparse
import os
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from easydict import EasyDict as edict

def prepare_arguments(cfg):
    '''
    get input arguments
    :return: args
    '''
    # args get from hydra -> easydict
    args = OmegaConf.to_container(cfg) if isinstance(cfg, DictConfig) else cfg
    args = edict(args)
    args = configure_exp_id(args)

    # save paths config & make dir
    args.checkpoint_path = os.path.join(args.checkpoint_path, f"{args.exp_id}")
    args.log_path = os.path.join(args.log_path, f"{args.exp_id}")
    args.output_path = os.path.join(args.output_path, f"{args.exp_id}")
    args.plot_path = os.path.join(args.plot_path, f"{args.exp_id}")
    args.result_path = os.path.join(args.result_path, f"{args.exp_id}")

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    args.home_dir = "."
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def configure_exp_id(args):
    '''
    Configure exp_id for hyperparameters manually.
    subject to change when wandb supports yaml variable references feature:
    https://github.com/wandb/wandb/issues/3707
    '''
    if args.exp_id == "default":
        model = args.model.name
        dataset = args.dataset
        new_exp_id = f"{model}_{dataset}_RevIN_{args.RevIN}"
        args.exp_id = new_exp_id
    return args


def EDA_prep_arguments(window_size=96, dataset="SWaT", scaler="std"):
    return edict({
        "home_dir": ".",
        "window_size": window_size,
        "stride": 1,
        "dataset": dataset,
        "batch_size": 64,
        "eval_batch_size": 64*3,
        "scaler": scaler,
        "window_anomaly": False,
    })