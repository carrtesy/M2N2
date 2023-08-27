######################################################
#                       _oo0oo_                      #
#                      o8888888o                     #
#                      88" . "88                     #
#                      (| -_- |)                     #
#                      0\  =  /0                     #
#                    ___/`---'\___                   #
#                  .' \\|     |// '.                 #
#                 / \\|||  :  |||// \                #
#                / _||||| -:- |||||- \               #
#               |   | \\\  -  /// |   |              #
#               | \_|  ''\---/''  |_/ |              #
#               \  .-\__  '-'  ___/-. /              #
#             ___'. .'  /--.--\  `. .'___            #
#          ."" '<  `.___\_<|>_/___.' >' "".          #
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |        #
#         \  \ `_.   \_ __\ /__ _/   .-` /  /        #
#     =====`-.____`.___ \_____/___.-`___.-'=====     #
#                       `=---='                      #
#                                                    #
#                                                    #
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    #
#                                                    #
#        Buddha Bless:   "No Bugs in my code"        #
#                                                    #
######################################################
import torch

import wandb
import hydra
from omegaconf import DictConfig

from utils.logger import make_logger
from utils.argpass import prepare_arguments
from utils.tools import SEED_everything
from utils.secret import WANDB_API_KEY

import warnings
import os
from data.load_data import DataFactory

from Exp.MLP import MLP_Tester
from Exp.LSTMEncDec import LSTMEncDec_Tester
from Exp.USAD import USAD_Tester
from Exp.THOC import THOC_Tester
from Exp.OmniAnomaly import OmniAnomaly_Tester
from Exp.AnomalyTransformer import AnomalyTransformer_Tester

import pandas as pd
from vus.utils.slidingWindows import find_length
from ast import literal_eval
import json


torch.set_num_threads(1)

warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="cfgs", config_name="test_defaults")
def main(cfg: DictConfig) -> None:

    # prepare arguments
    args = prepare_arguments(cfg)

    # WANDB
    #wandb.login(key=WANDB_API_KEY)
    WANDB_PROJECT_NAME, WANDB_ENTITY = "OnlineTSAD", "carrtesy"
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id, mode="offline")
    wandb.config.update(args)

    # Logger
    logger = make_logger(os.path.join(args.log_path, f'{args.exp_id}_test.log'))
    logger.info("=== TESTING START ===")
    logger.info(f"Configurations: {args}")

    # SEED
    SEED_everything(args.SEED)
    logger.info(f"Experiment with SEED: {args.SEED}")

    # Data
    logger.info(f"Preparing {args.dataset} dataset...")
    datafactory = DataFactory(args, logger)
    train_dataset, train_loader, test_dataset, test_loader = datafactory()
    args.num_channels = train_dataset.X.shape[1]

    # sliding window estimate for range-based metrics
    sliding_windows = [find_length(train_dataset.X[:, c]) for c in range(args.num_channels)]
    args.range_window_size = max(sliding_windows)
    logger.info(f"sliding window for estimating range-based metrics: {args.range_window_size}")

    # Model
    logger.info(f"Loading pre-trained {args.model.name} model...")
    Testers = {
        "MLP": MLP_Tester,
        "LSTMEncDec": LSTMEncDec_Tester,
        "USAD": USAD_Tester,
        "OmniAnomaly": OmniAnomaly_Tester,
        "THOC": THOC_Tester,
        "AnomalyTransformer": AnomalyTransformer_Tester,
    }

    tester = Testers[args.model.name](
        args=args,
        logger=logger,
        train_loader=train_loader,
        test_loader=test_loader,
        load=True,
    )

    # infer
    cols = ["tau", "Accuracy", "Precision", "Recall", "F1",  "tn", "fp", "fn", "tp"]
    cols += ["Accuracy_PA", "Precision_PA", "Recall_PA", "F1_PA", "tn_PA", "fp_PA", "fn_PA", "tp_PA"]
    cols += ["ROC_AUC", "PR_AUC", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]
    result_df = pd.DataFrame([], columns=cols)
    for option in args.infer_options:
        result = tester.infer(mode=option, cols=cols)
        result_df = pd.concat([result_df, result])

    logger.info(f"\n{result_df.to_string()}")

    # log result
    wt = wandb.Table(dataframe=result_df)
    wandb.log({"result_table": wt})

if __name__ == "__main__":
    main()