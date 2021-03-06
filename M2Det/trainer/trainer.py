import numpy as np
import torch
import wandb
from base import BaseTrainer
from utils import inf_loop, MetricTracker, get_lr
from torch.autograd import Variable


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.init_lr = config['optimizer']['args']['lr']
        self.warm_up = config['trainer']['warm_up']

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            # image, target = Variable(data['data'].float().cuda()), Variable(data['bbox'].float().cuda())
            # print(type(data['data']))
            # print(type(data['bbox'][0]))
            image = Variable(data['data'].float().cuda())
            with torch.no_grad():
                target = [Variable(i.float().cuda()) for i in data['bbox']]

            # Linear Learning Rate Warm-up
            full_batch_idx = ((epoch-1)*len(self.data_loader) + batch_idx)
            if epoch - 1 < self.warm_up:
                for params in self.optimizer.param_groups:
                    params['lr'] = self.init_lr/(self.warm_up * len(self.data_loader)) * full_batch_idx
            lr = get_lr(self.optimizer)

            # -------- TRAINING LOOP --------
            self.optimizer.zero_grad()
            output = self.model(image)
            reg_loss, cls_loss = self.criterion(output, target)
            loss = reg_loss + cls_loss
            loss.backward()
            self.optimizer.step()
            # -------------------------------

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()

            # self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

            # log = self.train_metrics.result()
            wandb_log = {'loss': loss.item(),
                         'cls_loss': cls_loss.item(),
                         'reg_loss': reg_loss.item()}

            # Add log to WandB
            if not self.config['debug']:
                wandb.log(wandb_log)

        # log = self.train_metrics.result()
        log = {'loss': total_loss / self.len_epoch,
               'cls_loss': total_cls_loss / self.len_epoch,
               'reg_loss': total_reg_loss / self.len_epoch}

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        log.update({'lr': lr})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
