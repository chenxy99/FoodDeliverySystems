import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import models.math_util as math_util
from models.sample_util import SampleEpisode
from utils.recording import RecordManager
from utils.checkpointing import CheckpointManager


epsilon = 1e-8

class ActorCritic(nn.Module):
    def __init__(self, args, log_dir, checkpoints_dir):
        super(ActorCritic, self).__init__()
        self.args = args
        self.log_dir = log_dir
        self.checkpoints_dir = checkpoints_dir
        self.current_metric = 0
        self.current_eposide = 0.0
        self.duration = 0
        self.actor = Actor(args).cuda()
        self.critic = Critic(args).cuda()


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.alpha_a, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=args.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.alpha_c, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=args.weight_decay)
        if self.args.eval:
            self.load_optimal_parameters()
        else:
            self.initialize()

        self.lr_actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer,
                                                         lr_lambda=self.lr_lambda, last_epoch=self.iteration)
        self.lr_critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer,
                                                          lr_lambda=self.lr_lambda, last_epoch=self.iteration)

    def lr_lambda(self, iteration):
        if iteration <= self.args.episode * self.args.warmup_percent:
            return iteration / (self.args.episode * self.args.warmup_percent)
        else:
            return 1 - (iteration - self.args.episode * self.args.warmup_percent) / \
                   (self.args.episode * (1.0 - self.args.warmup_percent))


    def one_iteration(self):
        # if self.args.eval:
        #     self.duration = self.args.duration
        # else:
        #     self.duration = int(self.current_eposide / float(self.args.episode) * self.args.duration) + 2
        self.duration = self.args.duration
        batch = SampleEpisode(self, duration=self.duration)
        # for testing
        # batch = dict()
        # batch["states"] = torch.randn((self.args.duration + 1, self.args.num_people * 3 + 3,
        #                                self.args.city_size, self.args.city_size))
        # batch["rewards"] = torch.randn((self.args.duration, 1))
        batch["dones"] = torch.ones((self.duration, 1))
        # pickup = torch.randint(low=0, high=self.args.num_people + 1, size=(self.args.duration, 1))
        # batch["pickup_controls"] = torch.zeros(self.args.duration, self.args.num_people+1).long()
        # batch["pickup_controls"].scatter_(dim=1, index=pickup,
        #                                   src=torch.ones(self.args.duration, 1).long())
        # action = torch.randint(low=0, high=4, size=(self.args.num_people, self.args.duration, 1))
        # batch["people_actions"] = torch.zeros(self.args.num_people, self.args.duration, 4).long()
        # batch["people_actions"].scatter_(dim=2, index=action,
        #                                  src=torch.ones(self.args.num_people, self.args.duration, 1).long())

        for key, value in batch.items():
            batch[key] = value.cuda()

        assert batch["states"].size() == (self.duration + 1, self.args.num_people * 3 + 3,
                                           self.args.city_size, self.args.city_size)
        assert batch["rewards"].size() == (self.duration, 1)
        assert batch["dones"].size() == (self.duration, 1)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # get the predictions of each states from critic excluding the last state
        v_preds = self.critic(batch["states"][:-1])
        # get the advantage estimations and the target value function
        advs, v_targets = self.calc_GAE_advs_v_target(batch, v_preds)
        # get the policy of each actions given a specific state
        pickup_controls, people_actions = self.actor(batch["states"][:-1])
        # calcuate the entropy of each policy
        H_pickup_controls = math_util.entropy(F.softmax(pickup_controls, -1))
        H_people_actions = [math_util.entropy(F.softmax(x, -1)) for x in people_actions]
        H_people_actions = torch.stack(H_people_actions, dim=0)
        H_people_actions = H_people_actions.sum(0)
        # get the logprob of the policy
        logprob_pickup_controls = F.log_softmax(pickup_controls, -1)[batch["pickup_controls"]==1]
        people_actions_list = [F.log_softmax(people_actions[i], -1)[batch["people_actions"][i]==1] for i in range(self.args.num_people)]
        logprob_people_actions = torch.stack(people_actions_list, dim=0)
        logprob_people_actions = logprob_people_actions.sum(0)

        if (batch["pickup_controls"] == 1).sum() == 0:
            logprob_pickup_controls = logprob_people_actions.data * 0
        loss_value = F.mse_loss(v_preds, v_targets, reduction="mean")
        loss_policy = - (logprob_pickup_controls.mean() + logprob_people_actions.mean() +
                         self.args.beta * H_pickup_controls.mean() + self.args.beta * H_people_actions.mean()
        )

        loss = loss_value + loss_policy
        loss.backward()
        if self.args.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clip)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.current_metric = self.args.smooth_factor * self.current_metric + \
                              (1 - self.args.smooth_factor) * v_targets.mean().cpu().numpy()


    def train(self):
        with tqdm(total=self.args.episode, initial=self.iteration + 1) as pbar:
            for eposide in range(self.iteration, self.args.episode):
                self.current_eposide = eposide
                self.one_iteration()
                self.tensorboard_writer.add_scalar("reward", self.current_metric, eposide)
                self.tensorboard_writer.add_scalar("duration", self.duration, eposide)
                self.tensorboard_writer.add_scalar("actor_learning_rate", self.actor_optimizer.param_groups[0]["lr"],
                                                   eposide)
                self.tensorboard_writer.add_scalar("critic_learning_rate", self.critic_optimizer.param_groups[0]["lr"],
                                                   eposide)

                if eposide % self.args.log_every == 0 and eposide > 0:
                    # save
                    self.actor_checkpoint_manager.step(self.current_metric)
                    self.critic_checkpoint_manager.step(self.current_metric)
                    self.best_metric = self.actor_checkpoint_manager.get_best_metric()
                    self.record_manager.save(0, eposide, self.best_metric, self.current_metric)
                self.lr_actor_scheduler.step()
                self.lr_critic_scheduler.step()
                pbar.update(1)

    def evaluation(self):
        if self.args.eval:
            self.duration = self.args.duration
        else:
            self.duration = int(self.current_eposide / float(self.args.episode) * self.args.duration) + 2
        with torch.no_grad():
            batch = SampleEpisode(self, duration=self.duration)
            # for testing
            batch["dones"] = torch.ones((self.duration, 1))

            for key, value in batch.items():
                batch[key] = value.cuda()

            assert batch["states"].size() == (self.duration + 1, self.args.num_people * 3 + 3,
                                               self.args.city_size, self.args.city_size)
            assert batch["rewards"].size() == (self.duration, 1)
            assert batch["dones"].size() == (self.duration, 1)

            # get the predictions of each states from critic excluding the last state
            v_preds = self.critic(batch["states"][:-1])

            print("The average award function is :{v_preds}".format(v_preds=v_preds.mean()))


    def calc_GAE_advs_v_target(self, batch, v_preds):
        '''
        Calculate GAE, and advs = GAE, v_targets = GAE + v_pred[-1]
        :param batch:
        :param v_preds: the value function of all the T states, excluding the last states
        :return advs: advantage estimation
                v_targets: value function target
        '''
        v_pred_detach = v_preds.detach()
        with torch.no_grad():
            last_pred = self.critic(batch["states"][-1].unsqueeze(0))
        v_all_preds = torch.cat([v_pred_detach, last_pred], dim=0)
        gaes = math_util.calc_GAEs(batch["rewards"], batch["dones"], v_all_preds, self.args.gamma, self.args.lam)
        v_targets = gaes + v_pred_detach[-1]
        advs = math_util.standardize(v_targets)
        return advs, v_targets

    def initialize(self):
        # --------------------------------------------------------------------------------------------
        #  BEFORE TRAINING STARTS
        # --------------------------------------------------------------------------------------------

        # Tensorboard summary writer for logging losses and metrics.
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)

        # Record manager for writing and loading the best metrics and theirs corresponding epoch
        self.record_manager = RecordManager(self.log_dir)
        if self.args.resume_dir == '':
            self.record_manager.init_record()
        else:
            self.record_manager.load()

        self.start_epoch = self.record_manager.get_epoch()
        self.iteration = self.record_manager.get_iteration()
        self.best_metric = self.record_manager.get_best_metric()
        self.current_metric = self.record_manager.get_current_metric()

        # Checkpoint manager to serialize checkpoints periodically while training and keep track of
        # best performing checkpoint.
        self.actor_checkpoint_manager = CheckpointManager(self.actor, self.actor_optimizer, self.checkpoints_dir,
                                                          mode="max", best_metric=self.best_metric,
                                                          filename_prefix="actor_checkpoint")
        self.critic_checkpoint_manager = CheckpointManager(self.critic, self.critic_optimizer, self.checkpoints_dir,
                                                           mode="max", best_metric=self.best_metric,
                                                           filename_prefix="critic_checkpoint")

        # Load checkpoint to resume training from there if specified.
        # Infer iteration number through file name (it's hacky but very simple), so don't rename
        # saved checkpoints if you intend to continue training.
        if self.args.resume_dir != "":
            training_checkpoint = torch.load(os.path.join(self.checkpoints_dir, "actor_checkpoint.pth"))
            for key in training_checkpoint:
                if key == "optimizer":
                    self.actor_optimizer.load_state_dict(training_checkpoint[key])
                else:
                    self.actor.load_state_dict(training_checkpoint[key])
            training_checkpoint = torch.load(os.path.join(self.checkpoints_dir, "critic_checkpoint.pth"))
            for key in training_checkpoint:
                if key == "optimizer":
                    self.critic_optimizer.load_state_dict(training_checkpoint[key])
                else:
                    self.critic.load_state_dict(training_checkpoint[key])

    def load_optimal_parameters(self):
        # Load checkpoint to start evaluation.
        # Infer iteration number through file name (it's hacky but very simple), so don't rename
        test_checkpoint = torch.load(os.path.join(self.checkpoints_dir, "actor_checkpoint_best.pth"))
        for key in test_checkpoint:
            if key == "optimizer":
                continue
            else:
                self.actor.load_state_dict(test_checkpoint[key], strict=False)
        test_checkpoint = torch.load(os.path.join(self.checkpoints_dir, "critic_checkpoint_best.pth"))
        for key in test_checkpoint:
            if key == "optimizer":
                continue
            else:
                self.critic.load_state_dict(test_checkpoint[key], strict=False)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.ratio = 16
        self.input = self.args.num_people * 3 + 3
        self.conv_1 = nn.Conv2d(self.input, 32, kernel_size=3, padding=1, stride=2, bias=True)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=True)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=True)
        # self.conv_5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=True)
        self.pickup_control = nn.Linear(128 * (self.args.city_size // self.ratio) ** 2, self.args.num_people + 1)
        self.people_actions = nn.ModuleList([nn.Linear(128 * (self.args.city_size // self.ratio) ** 2, 4)
                                             for _ in range(self.args.num_people)])

        self.init_weights()

    def forward(self, input):
        x = F.relu(self.conv_1(input))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        # x = F.relu(self.conv_5(x))
        x = x.reshape(x.shape[0], -1)
        pickup_controls = self.pickup_control(x)
        people_actions = [self.people_actions[i](x) for i in range(self.args.num_people)]

        return pickup_controls, people_actions

    def init_weights(self):
        # Common practise for initialization.
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val=1.0)
                torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.ratio = 16
        self.input = self.args.num_people * 3 + 3
        self.conv_1 = nn.Conv2d(self.input, 32, kernel_size=3, padding=1, stride=2, bias=True)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=True)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=True)
        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=True)
        # self.conv_5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=True)
        self.fc = nn.Linear(128 * (self.args.city_size // self.ratio) ** 2, 1)

    def forward(self, input):
        x = F.relu(self.conv_1(input))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        # x = F.relu(self.conv_5(x))
        y = self.fc(x.reshape(x.shape[0], -1))

        return y

    def init_weights(self):
        # Common practise for initialization.
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val=1.0)
                torch.nn.init.constant_(layer.bias, val=0.0)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
