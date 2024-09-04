# run_example.py
import os
import sys
import torch
import logging
import argparse
import traceback
import numpy as np
from pprint import pprint

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset

from utils.logger import setup_logging
from utils.arg_helper import get_config, get_config_
from src.MSG_model.msg_convnet import MSGConvNet
from runner.pyg_runner import PYGRunner

def main():
    # 1. Parser
    parser = argparse.ArgumentParser(
        description="Running experiment of graph classification"
    )
    parser.add_argument('--script_cfg', type=str, default='config/DEFAULT/DEF_config.yaml',
                        help='Script config file path')
    parser.add_argument('--MSGConv_cfg', type=str, default='config/DEFAULT/DEF_msgcfg.yaml',
                        help='MSGConv config file path')
    parser.add_argument('--NeuralNet_cfg', type=str, default='config/DEFAULT/DEF_nncfg.yaml',
                        help='NeuralNet config file path')
    parser.add_argument('--use_gdc', type=str, default='False')
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='A TUDataset. The task at hand must be graph classification.')

    parser.add_argument('--log_level', type=str, default='INFO',
                        help="Logging Level, \
                          DEBUG, \
                          INFO, \
                          WARNING, \
                          ERROR, \
                          CRITICAL")
    parser.add_argument('--comment', type=str, help="Experiment comment")
    parser.add_argument('--test', type=str, default='False')
    args = parser.parse_args()

    # 2. Load model
    script_cfg = get_config(args.script_cfg)
    MSGConv_cfg = get_config_(args.MSGConv_cfg)
    NeuralNet_cfg = get_config_(args.NeuralNet_cfg)
    model = MSGConvNet(MSGConv_cfg, NeuralNet_cfg)

    # 4. Dataset
    dataset = TUDataset(root='/tmp/'+args.dataset, name=args.dataset)

    # 3. G.D.C. transform

    torch.manual_seed(script_cfg.seed)
    torch.cuda.manual_seed_all(script_cfg.seed)
    dataset = dataset.shuffle()
    train_dataset = dataset[:-223]    # hardcoding
    dev_dataset = dataset[-223:]

    # 5. Logger
    log_file = os.path.join(script_cfg.save_dir,
                            "log_exp_{}.txt".format(script_cfg.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(script_cfg.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint(script_cfg)
    print("<" * 80)

    # 6. Runner
    script_cfg.use_gpu = script_cfg.use_gpu and torch.cuda.is_available()
    try:
      runner = PYGRunner(model_object=model, script_cfg=script_cfg,
                         train_dataset=train_dataset, dev_dataset=dev_dataset)
      runner.train()

    except:
      logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()

