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

import wandb
import hydra
from omegaconf import DictConfig

from utils.logger import make_logger
from utils.argpass import prepare_arguments
from utils.tools import SEED_everything

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
import torch

# Computational Cost
from fvcore.nn import FlopCountAnalysis
from utils.memory_cost_profiler import profile_memory_cost
import numpy as np
#from torchsummary import summary
from torchinfo import summary

warnings.filterwarnings("ignore")

import json
@hydra.main(version_base=None, config_path="cfgs", config_name="test_defaults")
def main(cfg: DictConfig) -> None:
    # SEED
    SEED_everything(2023)

    # prepare arguments
    args = prepare_arguments(cfg)

    # WANDB
    wandb.login()
    WANDB_PROJECT_NAME, WANDB_ENTITY = "OnlineTSAD", "carrtesy"
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id)
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
    )

    ip_shape = (1, args.window_size, args.num_channels) if args.model.name != "USAD" else (1, args.window_size*args.num_channels)
    logger.info(ip_shape)
    dummy_input = torch.randn(ip_shape).to(args.device)

    # Time Cost
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = tester.model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = tester.model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    # Flops
    flops = FlopCountAnalysis(tester.model, dummy_input)

    # parameter size
    pcnt = sum(p.numel() for p in tester.model.parameters() if p.requires_grad)

    # final outputs
    logger.info(f"mean:{mean_syn}, std:{std_syn}")
    logger.info(f"total flops: {flops.total()}")
    logger.info(f"psize: {pcnt}")

    logger.info("=== with forward backward both ===")
    if args.model.name =="USAD":
        logger.info(summary(tester.model, input_size=(1, args.window_size*args.num_channels)))
    else:
        logger.info(summary(tester.model, input_size=(1, args.window_size, args.num_channels)))

    with open(f'./{args.model.name}.json', 'w') as fp:
        data = {
            "total_flops": flops.total(),
            "time": mean_syn,
            "time_std": std_syn,
            "psize": pcnt,
        }
        # memory cost
        if args.model.name == "MLP":
            # memory_cost, {'param_size': param_size, 'act_size': activation_size}

            memory_cost, cost_comp = profile_memory_cost(tester.model, input_size=ip_shape, batch_size=1)
            data["memory_cost"] = memory_cost
            data.update(cost_comp)
            logger.info(f"memory_cost: {memory_cost}")
            param_size = data["param_size"]
            logger.info(f"param_size: {param_size}")

            logger.info("=== before ===")
            logger.info(f"memory_cost: {memory_cost}")
            logger.info(f"cost_comp: {cost_comp}")

            for param in tester.model.parameters():
                param.requires_grad = False

            logger.info("=== after ===")
            memory_cost_f, cost_comp_f = profile_memory_cost(tester.model, input_size=ip_shape, batch_size=1)
            logger.info(f"memory_cost_f: {memory_cost_f}")
            logger.info(f"cost_comp_f: {cost_comp_f}")


        json.dump(data, fp)


    # logger.info("=== with forward only ===")
    # for param in tester.model.parameters():
    #     param.requires_grad = False
    # if args.model.name =="USAD":
    #     logger.info(summary(tester.model, input_size=(1, args.window_size*args.num_channels)))
    # else:
    #     logger.info(summary(tester.model, input_size=(1, args.window_size, args.num_channels)))


if __name__ == "__main__":
    main()