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
        new_exp_id = f"{model}_{dataset}"
        if model == "OCSVM":
            new_exp_id += f"_nu_{args.model.nu}"
        elif model == "IsolationForest":
            new_exp_id += f"_n_estimator_{args.model.n_estimators}"
            new_exp_id += f"_contamination_{args.model.contamination}"
        elif model == "LOF":
            new_exp_id += f"_contamination_{args.model.contamination}"
        elif model == "LSTMEncDec":
            new_exp_id += f"_latent_dim_{args.model.latent_dim}"
            new_exp_id += f"_num_layers_{args.model.num_layers}"
            new_exp_id += f"_dropout_{args.model.dropout}"
        elif model == "LSTMVAE":
            new_exp_id += f"_hidden_dim_{args.model.hidden_dim}"
            new_exp_id += f"_z_dim_{args.model.z_dim}"
            new_exp_id += f"_n_layers_{args.model.n_layers}"
            new_exp_id += f"_beta_{args.model.beta}"
        elif model == "USAD":
            new_exp_id += f"_latent_dim_{args.model.latent_dim}"
            new_exp_id += f"_alpha_{args.model.alpha}"
            new_exp_id += f"_beta_{args.model.beta}"
            new_exp_id += f"_dsr_{args.model.dsr}"
        elif model == "OmniAnomaly":
            new_exp_id += f"_hidden_dim_{args.model.hidden_dim}"
            new_exp_id += f"_z_dim_{args.model.z_dim}"
            new_exp_id += f"_dense_dim_{args.model.dense_dim}"
            new_exp_id += f"_beta_{args.model.beta}"
        elif model == "DeepSVDD":
            pass
        elif model == "DAGMM":
            new_exp_id += f"_gmm_k_{args.model.gmm_k}"
            new_exp_id += f"_latent_dim_{args.model.latent_dim}"
            new_exp_id += f"_lambda_energy_{args.model.lambda_energy}"
            new_exp_id += f"_lambda_cov_diag_{args.model.lambda_cov_diag}"
            new_exp_id += f"_grad_clip_{args.model.grad_clip}"
        elif model == "THOC":
            new_exp_id += f"_hidden_dim_{args.model.hidden_dim}"
            new_exp_id += f"_L2_reg_{args.model.L2_reg}"
            new_exp_id += f"_LAMBDA_orth_{args.model.LAMBDA_orth}"
            new_exp_id += f"_LAMBDA_TSS_{args.model.LAMBDA_TSS}"
        elif model == "AnomalyTransformer":
            new_exp_id += f"_anomaly_ratio_{args.model.anomaly_ratio}"
            new_exp_id += f"_k_{args.model.k}"
            new_exp_id += f"_temperature_{args.model.temperature}"
            new_exp_id += f"_e_layers_{args.model.e_layers}"
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