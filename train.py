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

import os
import wandb
import hydra
from omegaconf import DictConfig
import warnings; warnings.filterwarnings("ignore")

from utils.logger import make_logger
from utils.argpass import prepare_arguments
from utils.tools import SEED_everything
from utils.secret import WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT_NAME

import torch

from Exp import MLP_Trainer, LSTMEncDec_Trainer, USAD_Trainer, THOC_Trainer, AnomalyTransformer_Trainer
from data.load_data import DataFactory


@hydra.main(version_base=None, config_path="cfgs", config_name="train_defaults")
def main(cfg: DictConfig) -> None:
    # prepare arguments
    args = prepare_arguments(cfg)

    # WANDB
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id)
    wandb.config.update(args)

    # Logger
    logger = make_logger(os.path.join(args.log_path, f'{args.exp_id}_train.log'))
    logger.info("=== TRAINING START ===")
    logger.info(f"Configurations: {args}")

    # SEED
    SEED_everything(args.SEED)
    logger.info(f"Experiment with SEED: {args.SEED}")

    # Data
    logger.info(f"Preparing {args.dataset} dataset...")
    datafactory = DataFactory(args, logger)
    train_dataset, train_loader, test_dataset, test_loader = datafactory()
    args.num_channels = train_dataset.X.shape[1]

    # Model
    logger.info(f"Preparing {args.model.name} Trainer...")
    Trainers = {
        "MLP": MLP_Trainer,
        "LSTMEncDec": LSTMEncDec_Trainer,
        "USAD": USAD_Trainer,
        "THOC": THOC_Trainer,
        "AnomalyTransformer": AnomalyTransformer_Trainer,
    }

    trainer = Trainers[args.model.name](
        args=args,
        logger=logger,
        train_loader=train_loader,
    )

    # 4. train
    logger.info(f"Preparing {args.model.name} Training...")
    trainer.train()
    logger.info("=== TRAINING FINISHED ===")


if __name__ == "__main__":
    main()