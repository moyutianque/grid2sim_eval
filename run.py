#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
# import vlnce_baselines  # noqa: F401
from config.default import get_config
from agent.nonlearning_agents import (
    evaluate_agent,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))

def run_exp(exp_config: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    
    torch.backends.cudnn.deterministic = True
    if config.EVAL.EVAL_NONLEARNING:
        evaluate_agent(config)
        return


if __name__ == "__main__":
    main()
