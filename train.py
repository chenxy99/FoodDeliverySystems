import os
import argparse
import datetime
import json
from tqdm import tqdm

import torch
import numpy as np

from utils.logger import Logger
from models.actor_critic import ActorCritic

parser = argparse.ArgumentParser(description="Food delivery systems")
parser.add_argument("--eval", type=bool, default=False, help="Determine the mode")
parser.add_argument("--log_root", type=str, default="./assets/nsteps/person", help="Log root")
parser.add_argument("--resume_dir", type=str, default="", help="Resume from a specific directory")
parser.add_argument("--city_size", type=int, default=16, help="Size of the city map")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
parser.add_argument("--clip", type=float, default=12.5, help="Gradient clipping")
parser.add_argument("--log_every", type=int, default=100, help="Log step")
parser.add_argument("--smooth_factor", type=float, default=0.6, help="Smooth factor")
parser.add_argument("--warmup_percent", type=float, default=0.1, help="The portion of the warmup iteration")

# Actor
parser.add_argument("--beta", type=float, default=0.001, help="Entropy regularization weight")
parser.add_argument("--alpha_a", type=float, default=5e-6, help="Actor learning rate")

# Critic
parser.add_argument("--alpha_c", type=float, default=5e-6, help="Critic learning rate")

# Agent
parser.add_argument("--gae", type=bool, default=True, help="Whether to use GAE")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lam", type=float, default=0.97, help="Soft choice for GAE")
parser.add_argument("--duration", type=int, default=400, help="Length of the samples (s_t, a_t, r_t, s_t')")
parser.add_argument("--episode", type=int, default=10000, help="Number of the total episodes")
parser.add_argument("--num_people", type=int, default=3, help="Number of people")

args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():
    # setup logger
    if args.resume_dir == "":
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        log_dir = os.path.join(args.log_root, "log_" + date)
    else:
        log_dir = args.resume_dir
    hparams_file = os.path.join(log_dir, "hparams.json")
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if args.resume_dir == "":
        # write hparams
        with open(hparams_file, "w") as f:
            json.dump(args.__dict__, f, indent=2)
    log_file = os.path.join(log_dir, "log_train.txt")
    logger = Logger(log_file)
    # logger.info(args)
    logger.info("The args corresponding to training process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    actor_critic = ActorCritic(args, log_dir, checkpoints_dir)
    actor_critic.train()

if __name__ == "__main__":
    main()

