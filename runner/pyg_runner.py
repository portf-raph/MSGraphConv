# Modified from https://github.com/lrjconan/LanczosNetwork/blob/master/utils/train_helper.py
# (MIT Licensed source)


import os
import numpy as np
import pickle

from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter

import logging
from utils.train_helper import *


logger = logging.getLogger('exp_logger')

class PYGRunner(object):
    def __init__(
        self,
        model_object: nn.Module,
        script_cfg: EasyDict,
        train_dataset: Dataset,
        dev_dataset: Dataset,
        test_dataset: Dataset=None,
        ):

        if not isinstance(script_cfg, EasyDict):
            raise TypeError("Script config file is not in EasyDict format")

        self.script_cfg = script_cfg
        self.train_cfg = script_cfg.train
        self.test_cfg = script_cfg.test
        self.use_gpu = script_cfg.use_gpu
        self.gpus = script_cfg.gpus
        self.writer = SummaryWriter(script_cfg.save_dir)

        self.model = model_object
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset


    def train(self):
        # data loaders
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.train_cfg.batch_size,
                                  shuffle=self.train_cfg.shuffle,
                                  num_workers=self.train_cfg.num_workers,
                                 )

        dev_loader = DataLoader(dataset=self.dev_dataset,
                                batch_size=self.train_cfg.batch_size,
                                num_workers=self.train_cfg.num_workers)

        # model
        model = self.model
        if self.use_gpu:
            device = torch.device('cuda')
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.train_cfg.optimizer == 'SGD':
          optimizer = optim.SGD(
              params,
              lr=self.train_cfg.lr,
              momentum=self.train_cfg.momentum,
              weight_decay=self.train_cfg.wd)

        elif self.train_cfg.optimizer == 'Adam':
          optimizer = optim.Adam(
              params,
              lr=self.train_cfg.lr,
              weight_decay=self.train_cfg.wd)
        else:
          raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=True)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_cfg.lr_decay_steps,
            gamma=self.train_cfg.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training: TODO import load_model from utils
        if self.train_cfg.is_resume:
            load_model(self.model, self.train_cfg.resume_model, optimizer=optimizer)

        # training loop
        iter_count = 0
        best_val_loss = np.inf
        results = defaultdict(list)
        for epoch in range(self.train_cfg.max_epoch):
            epoch_loss = []
            # validation
            if (epoch + 1) % self.train_cfg.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                correct = 0

                for data in tqdm(dev_loader):
                    if self.use_gpu:
                        data = data.to(device)
                    with torch.no_grad():
                        out = self.model(x=data.x,
                                         edge_index=data.edge_index,
                                         edge_weight=data.edge_attr,
                                         batch=data.batch)
                        curr_loss = F.nll_loss(out, data.y).cpu().numpy()
                        val_loss += [curr_loss]    # appending
                        pred = out.max(dim=1)[1]
                        correct += pred.eq(data.y).sum().item()

                val_acc = correct / len(dev_loader.dataset)
                val_loss = float(np.mean(val_loss))   # no concat (mod)
                print("Avg. Validation CrossEntropy = {}".format(val_loss))     # dbg
                print("Avg. Validation Accuracy = {}".format(val_acc))
                self.writer.add_scalar('val_loss', val_loss, iter_count)
                results['val_loss'] += [val_loss]

                # save best model
                if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  snapshot(
                      model.module if self.use_gpu else model,
                      optimizer,
                      self.script_cfg,
                      epoch + 1,
                      tag='best')

                logger.info("Current Best Validation CrossEntropy = {}".format(best_val_loss))

                # check early stop
                if early_stop.tick([val_acc]):
                  print("STOPPING TIME DUE NOW")
                  snapshot(
                      model.module if self.use_gpu else model,
                      optimizer,
                      self.script_cfg,
                      epoch + 1,
                      tag='last')
                  self.writer.close()
                  break

            # training
            model.train()
            lr_scheduler.step()
            for data in train_loader:
                optimizer.zero_grad()

                if self.use_gpu:
                    data = data.to_device()
                out = self.model(x=data.x,
                                 edge_index=data.edge_index,
                                 edge_weight=data.edge_attr,
                                 batch=data.batch)
                train_loss = F.nll_loss(out, data.y)
                train_loss.backward()
                optimizer.step()
                train_loss = float(train_loss.data.cpu().numpy())
                self.writer.add_scalar('train_loss', train_loss, iter_count)
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                epoch_loss += [train_loss]

                # display loss
                if (iter_count + 1) % self.train_cfg.display_iter == 0:
                    logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(   # dbg
                        epoch + 1, iter_count + 1, train_loss))

                iter_count += 1
            if (epoch + 1) % self.train_cfg.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module
                         if self.use_gpu else model, optimizer, self.script_cfg, epoch + 1)

            epoch_loss = float(np.mean(epoch_loss))
            print("Loss @ epoch {:04d} = {}".format(epoch+1, epoch_loss))

        results['best_val_loss'] += [best_val_loss]
        pickle.dump(results,
                    open(os.path.join(self.script_cfg.save_dir, 'train_stats.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation MSE = {}".format(best_val_loss))

        return best_val_loss


    def test(self):
        test_loader = DataLoader(dataset=self.dev_dataset,
                                 batch_size=self.test_cfg.batch_size,
                                 shuffle=False,
                                 num_workers=self.test_cfg.num_workers,
                                 drop_last=False)
        load_model(self.model, self.test_cfg.test_model)

        if self.use_gpu:
            device = torch.device('cuda')
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        test_loss = []
        for data in tqdm(test_loader):
            if self.use_gpu:
                data = data.to(device)
            with torch.no_grad():
                out = model(x=data.x,
                            edge_index=data.edge_index,
                            edge_weight=data.edge_attr,
                            batch=data.batch)
                curr_loss = F.nll_loss(out, data.y).cpu().numpy()
                test_loss += [curr_loss]

        test_loss = float(np.mean(np.concatenate(test_loss)))
        logger.info("Test MSE = {}".format(test_loss))

        return test_loss
