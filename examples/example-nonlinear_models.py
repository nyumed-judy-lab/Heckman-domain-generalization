
import os
import math
import copy
import time
import yaml
import typing
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import wandb

from rich import print as print
from rich.progress import track
from torch.utils.data import ConcatDataset, DataLoader

from configs.defaults import DefaultConfig

from datasets.base import MultipleDomainCollection
from datasets.samplers import configure_train_sampler

from models.base import _DeepModelBase
from models.heckman.multinomial_utils import MatrixOps

from networks.initializers import NetworkInitializer

from utils.calibration import draw_reliability_graph
from utils.callbacks import MetricAccumulator, ModelSelector
from utils.metrics import (MetricEvaluator,
                           expected_calibration_error,
                           maximum_calibration_error,
                           multilabel_accuracy,
                           multilabel_f1,
                           multilabel_precision,
                           multilabel_recall)
from utils.optimization import (configure_learning_rate_scheduler,
                                configure_optimizer)
from utils.transforms import InputTransforms


class _HeckmanDGBase(_DeepModelBase):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig) -> None:
        """
        Base class for HeckmanDG-1 type models.
        Arguments:
            args: (argparse.Namespace) for command line arguments
            defaults: (config.defaults.DefaultConfig) for data-specific settings
        """
        super(_HeckmanDGBase, self).__init__(args=args, defaults=defaults)

        self._normal = torch.distributions.Normal(loc=0., scale=1.)
        for s in ('train_domains', ):
            if not hasattr(self.defaults, s):
                raise AttributeError(f"{s} not found.")

    def _init_modules(self) -> None:
        raise NotImplementedError

    def _init_transforms(self) -> None:
        raise NotImplementedError

    def _init_cuda(self) -> None:
        raise NotImplementedError

    def _init_optimization(self) -> None:
        raise NotImplementedError

    def _init_checkpoint_dir(self) -> None:
        raise NotImplementedError

    @property
    def train_domains(self) -> typing.List[int]:
        return self.defaults.train_domains

    @property
    def validation_domains(self) -> typing.List[int]:
        return self.defaults.validation_domains

    @property
    def test_domains(self) -> typing.List[int]:
        return self.defaults.test_domains


class HeckmanDGPretrain(_HeckmanDGBase):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig):
        super(HeckmanDGPretrain, self).__init__(args=args, defaults=defaults)

    def _init_modules(self) -> None:

        # \varphi_{g}
        self.selection_encoder = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )

        # \omega_{g,k}, \forall{k} \in {1, ..., K}
        self.selection_head = nn.Linear(
            in_features=self.selection_encoder.out_features, out_features=len(self.train_domains),
        )

        self.logger.info(f"Initialized modules.")

        # (Optional) Load encoder weights from a checkpoint file
        if self.args.pretrained_model_file is not None:
            if os.path.isfile(self.args.pretrained_model_file):
                self.load_checkpoint(self.args.pretrained_model_file)
            else:
                self.logger.info(
                    f"Invalid pretrained model file. Proceeding without loading weights."
                )
        
        # (Optional) Freeze encoder weights
        if self.args.freeze_encoder:
            for p in self.selection_encoder.parameters():
                p.requires_grad = False
        
        self.logger.info(f"Freeze encoder weights: {self.args.freeze_encoder}")

    def _init_transforms(self) -> None:
        
        InputTransformObj: object = InputTransforms[self.args.data]
        self.eval_transform = InputTransformObj(augmentation=False)
        self.train_transform = InputTransformObj(
            augmentation=self.args.augmentation, randaugment=self.args.randaugment,
        )

        self.logger.info(f"Train transform = {self.train_transform}")
        self.logger.info(f"Eval transform = {self.eval_transform}")

    def _init_cuda(self) -> None:

        self.selection_encoder.to(self.device)
        self.selection_head.to(self.device)
        self.train_transform.to(self.device)
        self.eval_transform.to(self.device)

        self.logger.info(f"GPU: {self.device}")

    def _init_optimization(self) -> None:

        # optimizer
        self.optimizer = configure_optimizer(
            params=[
                {'params': self.selection_encoder.parameters()},  # Unnecessary if `self.args.freeze_encoder`
                {'params': self.selection_head.parameters()}
            ],
            name=self.args.optimizer,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.logger.info(f"Optimizer (name={self.optimizer.__class__.__name__})")

        # learning rate scheduler
        if self.args.scheduler is not None:
            self.scheduler = configure_learning_rate_scheduler(
                optimizer=self.optimizer,
                name=self.args.scheduler,
                epochs=self.args.epochs,
                warmup_epochs=self.args.scheduler_lr_warmup,
                min_lr=self.args.scheduler_min_lr,
            )
            self.logger.info(f"Scheduler (name={self.scheduler.__class__.__name__})")
        else:
            self.scheduler = None
            self.logger.info(f"No LR scheduler.")

    def _init_checkpoint_dir(self) -> None:

        # create checkpoint directory
        _ckpt_base: str = f"./checkpoints/HeckmanDG/pretrain/{self.args.data}/{self.args.backbone}/"
        if self.args.data in ('poverty', 'povertymap', ):
            self.checkpoint_dir = os.path.join(_ckpt_base, f"{self.defaults.fold}/{self.args.hash}")
        else:
            self.checkpoint_dir = os.path.join(_ckpt_base, f"{self.args.hash}")
        os.makedirs(self.checkpoint_dir, exist_ok=False)

        # save command-line arguments to a yaml file under the checkpoint directory
        _config_file = os.path.join(self.checkpoint_dir, 'configuration.yaml')
        with open(_config_file, 'w') as fp:
            yaml.dump(vars(self.args), fp, default_flow_style=False)

    def fit(self, dataset: MultipleDomainCollection) -> typing.Dict[str, MetricAccumulator]:

        # Get datasets. Note that we use ID validation.
        train_sets: typing.List[torch.utils.data.Dataset] = dataset.get_train_data(as_dict=False)
        id_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_id_validation_data(as_dict=False)
        ood_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_ood_validation_data(as_dict=False)  # or `None`

        # Configure train sampler
        train_sampler = configure_train_sampler(
            datasets=train_sets,
            use_domain=self.args.uniform_across_domains,
            use_target=self.args.uniform_across_targets,
        )

        _fit_kwargs: typing.Dict[str, object] = {
            'train_set': ConcatDataset(train_sets),
            'id_validation_set': ConcatDataset(id_validation_sets),
            'ood_validation_set': ConcatDataset(ood_validation_sets),
            'train_sampler': train_sampler,
        }

        return self._fit(**_fit_kwargs)  # TODO: add support for test set

    def _fit(self,
             train_set: torch.utils.data.Dataset,
             id_validation_set: torch.utils.data.Dataset,
             ood_validation_set: torch.utils.data.Dataset,
             train_sampler: torch.utils.data.Sampler = None,
             **kwargs, ) -> typing.Dict[str, MetricAccumulator]:

        # Instantiate train loader
        train_loader_configs = dict(
            batch_size=self.args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.args.prefetch_factor,
            num_workers=self.args.num_workers,
        )
        train_loader = DataLoader(train_set, **train_loader_configs)

        # Instantiate validation loader
        eval_loader_configs = dict(
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=self.args.eval_prefetch_factor,
            num_workers=self.args.eval_num_workers,
        )
        id_val_loader = DataLoader(id_validation_set, **eval_loader_configs)
        ood_val_loader = DataLoader(ood_validation_set, **eval_loader_configs)

        # Buffers to store metric values returned on every epoch
        train_metrics = MetricAccumulator(name='train')
        id_val_metrics = MetricAccumulator(name='id_val')
        ood_val_metrics = MetricAccumulator(name='ood_val')

        # Model selector (based on validation performance)
        assert hasattr(self.defaults, 'pretrain_model_selection_metric')
        model_selector = ModelSelector(
            metric=self.defaults.pretrain_model_selection_metric,
            patience=self.args.early_stopping,
        )
        self.logger.info(f"Choosing best selection model(s) based on `{model_selector.metric}`.")
        self.logger.info(f"Focal loss: {self.args.focal}")

        # Fit model (epoch = 1, 2, ..., self.args.epochs)
        _msg_fmt: str = f">{len(str(self.args.epochs))}"
        for epoch in range(self.args.epochs):

            # Check termination condition
            if model_selector.terminate:
                self.logger.info(f"Early stopping at epoch {epoch:,}.")
                break

            self.logger.info(f"LR = {self._get_current_lr():.6f}")

            # (Pre-)train
            train_history: dict = self.train(train_loader)
            train_metrics.update(train_history)
            msg: str = f"(  Train) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Evaluation (on the ID validation set)
            id_val_history: dict = self.evaluate(id_val_loader, epoch=epoch, suffix='ID')
            id_val_metrics.update(id_val_history)
            msg: str = f"( ID Val) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in id_val_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Evaluation (on the OOD validation set)
            ood_val_history: dict = self.evaluate(ood_val_loader, epoch=epoch, suffix='OOD')
            ood_val_metrics.update(ood_val_history)
            msg: str = f"(OOD Val) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in ood_val_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Update model selection status
            model_selector.update(
                value=id_val_history[model_selector.metric],  # same as using `self.defaults.model_selection_metric`
                model={
                    'selection_encoder': self.selection_encoder,
                    'selection_head': self.selection_head,
                },
                step=epoch,
                logger=self.logger,
            )
            # Logging (https://wandb.ai)
            log = dict()
            log.update(train_metrics[-1])
            log.update(id_val_metrics[-1])
            log.update(ood_val_metrics[-1])
            log.update({
                'misc/lr': self._get_current_lr(),
                'misc/best_epoch': model_selector.best_step,
                'misc/no_improvement': model_selector.no_improvement,
            })
            if False:
                ################################################
                # Logging (https://wandb.ai)
                log = dict()
                log.update(train_metrics[-1])
                log.update(id_val_metrics[-1])
                log.update(ood_val_metrics[-1])
                log.update({
                    'misc/lr': self._get_current_lr(),
                    'misc/best_epoch': model_selector.best_step,
                    'misc/no_improvement': model_selector.no_improvement,
                })
                # wandb.log(log, step=epoch)

            # Save current best model
            if model_selector.no_improvement == 0:
                ckpt: str = os.path.join(self.checkpoint_dir, f'ckpt.best.pth.tar')
                self.logger.info(f"Saving best checkpoint to {ckpt}")
                self.save_checkpoint(path=ckpt, epoch=epoch, history=log, train_domains=self.train_domains)

            # Periodically save current model
            if ((1 + epoch) % self.args.save_every) == 0:
                _save_fmt: str = f"0{len(str(self.args.epochs))}"  # leading zeros matching epochs
                ckpt: str = os.path.join(self.checkpoint_dir, f'ckpt.{epoch:{_save_fmt}}.pth.tar')
                self.logger.info(f"Saving intermediate checkpoint to: {ckpt}")
                self.save_checkpoint(path=ckpt, epoch=epoch, history=log, train_domains=self.train_domains)

        self.logger.info("Finished model training.")

        # Logging for best model.
        best_epoch: int = model_selector.best_step
        self.logger.info(f"Best selection model obtained at {best_epoch:,} (metric={model_selector.metric})")
        msg: str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics[best_epoch].items()])
        self.logger.info("[white]" + msg, extra={'markup': True})
        msg: str = " | ".join([f"{k}: {v:.4f}" for k, v in id_val_metrics[best_epoch].items()])
        self.logger.info("[red]" + msg, extra={'markup': True})

        # TODO: write metrics to csv file.
        train_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'train_metrics.csv'))
        id_val_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'id_val_metrics.csv'))
        ood_val_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'ood_val_metrics.csv'))

        return {
            'train': train_metrics,
            'id_val': id_val_metrics,
            'ood_val': ood_val_metrics,
        }

    def train(self,
              train_loader: torch.utils.data.DataLoader) -> typing.Dict[str, torch.FloatTensor]:

        self._set_learning_phase(train=True)
        
        steps_per_epoch: int = len(train_loader)
        losses = torch.zeros(steps_per_epoch, dtype=torch.float32, device=self.device)
        s_pred, s_true = list(), list()  # buffers to accumulate predictions and targets

        _pbar_kwargs = dict(
            total=steps_per_epoch,
            transient=True,
            description=f":fire: {self.__class__.__name__} (data={self.args.data}, backbone={self.args.backbone}, n.iter={steps_per_epoch:,})"
        )
        for i, batch in track(enumerate(train_loader), **_pbar_kwargs):

            # fetch data (only inputs and domain labels are required)
            domain = batch['domain'].to(self.device, non_blocking=True)  # (B,  ); true domain indicators
            x = batch['x'].to(self.device, non_blocking=True)            # (B, *)
            if self.train_transform is not None:
                x = self.train_transform(x)

            # refine domain indicators (target)
            # (B, K); B one-hot vectors
            domain = torch.tensor(domain) ########## added
            s_true_2d = domain.view(-1, 1).eq(torch.tensor(self.train_domains, dtype=torch.long, device=self.device).view(1, -1)).long()
            
            # s_true_2d = domain.view(-1, 1).eq(
            #     torch.tensor(self.train_domains, dtype=torch.long, device=self.device).view(1, -1)
            # ).long()                                         # (B, K); B one-hot vectors
            s_true_1d = s_true_2d.nonzero(as_tuple=True)[1]  # (B,  )
            assert len(s_true_2d) == len(s_true_1d), "Only supports data with non-overlapping domains."

            # forward (prediction; our loss functions expects probits)
            s_pred_in_probits = self.selection_head(self.selection_encoder(x))  # (B, K)

            # loss computation
            loss = self._pretrain_loss(s_pred_in_probits,
                                       s_true_1d,
                                       label_smoothing=self.args.label_smoothing,
                                       focal=self.args.focal)

            # backward & update
            self.optimizer.zero_grad()
            loss.backward()
            _ = self.optimizer.step()

            # accumulate {loss, predictions, targets} for further logging
            with torch.no_grad():
                losses[i] = loss.clone().detach()
                s_pred += [s_pred_in_probits.clone().detach()]
                s_true += [s_true_2d]

        # update learning rate (if `self.scheduler` exists)
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler.step()

        # once every epoch, compute metrics
        metrics: dict = {'loss': losses.mean()}
        with torch.no_grad():
            s_pred = self._normal.cdf(torch.cat(s_pred, dim=0))  # (N, K); in probs
            s_true = torch.cat(s_true, dim=0)                    # (N, K)
            # metrics['accuracy'] = multilabel_accuracy(s_pred, s_true, average='macro')
            # metrics['recall'] = multilabel_recall(s_pred, s_true, average='macro')
            # metrics['precision'] = multilabel_precision(s_pred, s_true, average='macro')
            # metrics['f1'] = multilabel_f1(s_pred, s_true, average='macro')
            metrics['accuracy'] = multilabel_accuracy(s_pred, s_true, average='macro')
            metrics['recall'] = multilabel_recall(s_pred, s_true, average='macro')
            metrics['precision'] = multilabel_precision(s_pred, s_true, average='macro')
            metrics['f1'] = multilabel_f1(s_pred, s_true, average='macro')

        return metrics  # dictionary with string keys and tensor values

    def _set_learning_phase(self, train: bool = True) -> None:
        if train:
            if self.args.freeze_encoder:
                self.selection_encoder.eval()
            else:
                self.selection_encoder.train()
            self.selection_head.train()
        else:
            self.selection_encoder.eval()
            self.selection_head.eval()

    @torch.no_grad()
    def evaluate(self,
                 data_loader: torch.utils.data.DataLoader,
                 epoch: int,
                 suffix: str = ' ',
                 ) -> typing.Dict[str, torch.FloatTensor]:
        """Evaluate performance of selection model."""

        self._set_learning_phase(train=False)
        s_pred, s_true = list(), list()
        domains = list()

        _pbar_kwargs = dict(
            total=len(data_loader),
            transient=True,
            description=f":rocket:({suffix}) selection model evaluation (data={self.args.data}, backbone={self.args.backbone})"
        )
        for _, batch in track(enumerate(data_loader), **_pbar_kwargs):

            # fetch data
            domain = batch['domain'].to(self.device, non_blocking=True)
            x = batch['x'].to(self.device, non_blocking=True)
            if self.eval_transform is not None:
                x = self.eval_transform(x)

            # forward; shape = (B, K)
            s_pred_in_probits = self.selection_head(self.selection_encoder(x))
            s_true_one_hot = domain.view(-1, 1).eq(
                torch.tensor(self.train_domains, dtype=torch.long, device=self.device).view(1, -1)
            ).long()  # (B, K)

            # accumulate
            domains += [domain]
            s_true += [s_true_one_hot]
            s_pred += [s_pred_in_probits.clone().detach()]

        domains = torch.cat(domains, dim=0)  # true domain indicators
        s_true = torch.cat(s_true, dim=0)    # (N, K); one-hot (long)
        s_pred = torch.cat(s_pred, dim=0)    # in probits \in \mathbb{R}
        s_probs = self._normal.cdf(s_pred)   # in probabilities \in [0, 1]
        
        metrics = dict()
        metrics['loss'] = F.binary_cross_entropy(s_probs, s_true.float(), reduction='mean')  # TODO: rename
        metrics['accuracy'] = multilabel_accuracy(s_probs, s_true, average='macro')
        metrics['recall'] = multilabel_recall(s_probs, s_true, average='macro')
        metrics['precision'] = multilabel_precision(s_probs, s_true, average='macro')
        metrics['f1'] = multilabel_f1(s_probs, s_true, average='macro')

        # TODO: plot
        if suffix == 'ID':
            self.plot_predictions(s_pred_in_probits=s_pred, domains=domains,
                                  epoch=epoch)

        return metrics  # dictionary

    def _pretrain_loss(self,
                       s_pred: torch.FloatTensor,  # 2d predictions
                       s_true: torch.LongTensor,   # 1d targets
                       label_smoothing: float = 0.0,
                       focal: bool = False,
                       alpha: typing.Optional[float] = 1.0,
                       gamma: typing.Optional[float] = 2.0, ) -> torch.FloatTensor:
        """
            Loss function used to pretrain domain selection models.
            Supports label smoothing and focal weighting.
        """

        assert (s_pred.ndim == 2) and (s_true.ndim == 1), "(N, K) and (N,  )"
        assert (s_pred.shape[0] == s_true.shape[0])
        assert (label_smoothing >= 0.0) and (label_smoothing < 0.5)

        # transform probits into probabilities \in [0, 1]. shape = (N, K)
        s_pred_in_probs = self._normal.cdf(s_pred)

        # 1d indicators -> 2d one-hot vectors
        s_true_2d = F.one_hot(s_true, num_classes=s_pred.shape[1]).float()

        # hard targets -> soft targets (with label smoothing)
        s_true_2d_soft = s_true_2d * (1.0 - label_smoothing) + (1.0 - s_true_2d) * label_smoothing
        
        if not focal:
            return F.binary_cross_entropy(s_pred_in_probs, s_true_2d_soft, weight=None)
                                          
        # compute focal weights (for selected; S = 1)
        with torch.no_grad():
            sel_mask = s_true_2d.ge(0.5)  # or .eq(1)
            sel_weights = (1. - s_pred_in_probs).masked_scatter_(
                mask=torch.logical_not(sel_mask), source=torch.zeros_like(s_pred),
            )

        # compute focal weights (for not selected; S = 0)
        with torch.no_grad():
            not_sel_mask = s_true_2d.lt(0.5)  # or .eq(0)
            not_sel_weights = (1. - (1. - s_pred_in_probs)).masked_scatter_(
                mask=torch.logical_not(not_sel_mask), source=torch.zeros_like(s_pred)
            )

        # aggregate focal weights
        weights = (sel_weights + not_sel_weights).clone().detach()
        weights = alpha * torch.pow(weights, gamma)

        return F.binary_cross_entropy(s_pred_in_probs, s_true_2d_soft, weight=weights)

    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, path: str, epoch: int, **kwargs) -> None:

        ckpt = {
            'selection_encoder': self.selection_encoder.state_dict(),   # TODO: rename to `encoder`
            'selection_head': self.selection_head.state_dict(),         # TODO: rename to `head`
            'optimizer': self.optimizer.state_dict(),
        }
        if hasattr(self, 'scheduler'):
            if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
                ckpt['scheduler'] = self.scheduler.state_dict()

        ckpt['epoch'] = epoch
        if kwargs:
            ckpt.update(kwargs);

        torch.save(ckpt, path)

    def load_checkpoint(self,
                        path: str,
                        encoder_keys: typing.Iterable[str] = ['encoder', 'selection_encoder'],
                        head_keys: typing.Iterable[str] = ['selection_head', ],
                        load_optimizer: bool = False,
                        load_scheduler: bool = False, ) -> None:

        ckpt = torch.load(path)
        self.logger.info(f"Loading weights from: {path}")

        # load encoder weights
        is_enc_loaded: bool = False
        for key in encoder_keys:
            try:
                self.selection_encoder.load_state_dict(ckpt[key])
                self.logger.info(f"Loaded encoder weights using key = `{key}`")
                is_enc_loaded = True
                break
            except KeyError as _:
                self.logger.info(f"Invalid key: `{key}`. Trying next key.")
                continue
        if not is_enc_loaded:
            self.logger.info(f"Failed to load encoder weights using keys from {encoder_keys}")
        
        # load head weights
        is_head_loaded: bool = False
        for key in head_keys:
            try:
                self.selection_head.load_state_dict(ckpt[key])
                self.logger.info(f"Loaded head weights using key = `{key}`")
                is_head_loaded = True
                break
            except KeyError as _:
                self.logger.info(f"Invalid key: `{key}`. Trying next key.")
                continue
        if not is_head_loaded:
            self.logger.info(f"Failed to load head weights using keys from {head_keys}")

        if load_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        
        if load_scheduler:
            if self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                except KeyError as _:
                    pass

    def plot_predictions(self,
                         s_pred_in_probits: torch.FloatTensor,
                         domains: torch.LongTensor,
                         epoch: int,
                         **kwargs, ) -> None:

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        assert s_pred_in_probits.ndim == 2, "(N, K)"
        assert s_pred_in_probits.shape[0] == domains.shape[0], "(N, )"
        N, K = s_pred_in_probits.shape

        # figure
        # fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig, axes = plt.subplots(K, 1, figsize=(12, 8))

        # create dataframe
        df = pd.DataFrame(s_pred_in_probits.detach().cpu().numpy())
        df.columns = [f"probits_{k}" for k in range(K)]
        df['domain'] = domains.detach().cpu().numpy()
        
        # plot histograms
        # for k in range(K):
        #     sns.histplot(data=df,
        #                  x=f'probits_{k}', hue='domain', palette=plt.cm.tab10,
        #                  bins=1000, stat='count', ax=axes[k])

        for k in range(K):
            sns.histplot(data=df,
                         x=f'probits_{k}', hue='domain', palette='tab10',
                         bins=1000, stat='count', ax=axes[k])

        for k, ax in enumerate(axes):
            
            # if k > 0:
            #     ax.sharex(axes[0])  # share x-axis
            #     ax.sharey(axes[0])  # share y-axis
            
            ax.grid(True, which='major', axis='x', alpha=.25)         # set grid
            ax.tick_params(which='major', axis='both', labelsize=12)  # set fontsize
            
            true_domain: int = self.train_domains[k]
            ax.set_xlabel(r"$\hat{g}_{k}(x)$" + f" (k={true_domain})", fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

        plt.suptitle(r"Histogram of $\hat{g}_{k}(x)$", fontsize=15)
        plt.tight_layout(pad=1.08)

        filepath: str = os.path.join(self.checkpoint_dir, f'plots/g_histogram_{epoch}.pdf')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath);

    def _adjust_learning_rate_cosine_decay(self,
                                           steps_per_epoch: int,
                                           global_step: int) -> float:
        """Adjust learning rate based on a half-cosine schedule."""

        max_steps: int = self.args.epochs * steps_per_peoch
        warmup_steps: int = self.args.scheduler_lr_warmup * steps_per_epoch
        base_lr: float = self.args.lr * (self.args.batch_size / 256.)

        if global_step < warmup_steps:
            lr = base_lr * (global_step / warmup_steps)
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * global_step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    
class HeckmanDG(_HeckmanDGBase):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig) -> None:
        super(HeckmanDG, self).__init__(args=args, defaults=defaults)

    def _init_modules(self):

        # 1-1. \varphi_{g}
        self.selection_encoder = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
        # 1-2. \omega_{g,k}, \forall{k} \in {1,...,K}
        self.selection_head = nn.Linear(
            in_features=self.selection_encoder.out_features, out_features=len(self.train_domains),
        )
        self.logger.info(f"(Selection) ImageNet weights: {self.args.pretrained}")

        # 2-1. \varphi_{f}
        self.outcome_encoder = NetworkInitializer.initialize_backbone(
                name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
        # 2-2. \omega_{f}
        self.outcome_head = nn.Linear(
            in_features=self.outcome_encoder.out_features,
            out_features=NetworkInitializer._out_features[self.args.data][0],
        )
        self.logger.info(f"(Outcome) ImageNet weights: {self.args.pretrained}")
        
        ##################################################################
        # 3-0. load selection weights (TODO: sanity check)
        if self.args.selection_pretrained_model_file is not None:
            if os.path.isfile(self.args.selection_pretrained_model_file):
                self.load_selection_weights_from_checkpoint(self.args.selection_pretrained_model_file)
            else:
                self.logger.info(
                    f"Invalid selection pretrained model file. Proceeding without loading weights."
                )
        ##################################################################
        # 3-1. (Optional) freeze selection model (TODO: sanity check)
        if self.args.freeze_selection_encoder:
            for p in self.selection_encoder.parameters():
                p.requires_grad = False
        
        if self.args.freeze_selection_head:
            for p in self.selection_head.parameters():
                p.requires_grad = False
        ##################################################################
        # 3-2. load outcome weights (TODO: sanity check)
        if self.args.outcome_pretrained_model_file is not None:
            if os.path.isfile(self.args.outcome_pretrained_model_file):
                self.load_outcome_weights_from_checkpoint(self.args.outcome_pretrained_model_file)
            else:
                self.logger.info(
                    f"Invalid outcome pretrained model file. Proceeding without loading weights."
                )
        ##################################################################
        # 3-3. (Optional) freeze outcome model (TODO: sanity check)
        if self.args.freeze_outcome_encoder:
            for p in self.outcome_encoder.parameters():
                p.requires_grad = False

        # 3-2. moving average model of outcome {encoder, head}
        if self.args.sma_start_iter > 0:

            with torch.no_grad():
                self.outcome_encoder_sma = copy.deepcopy(self.outcome_encoder)
                self.outcome_head_sma = copy.deepcopy(self.outcome_head)
                self.outcome_encoder_sma.eval()
                self.outcome_head_sma.eval()

            self.global_iter: int = 0
            self.sma_count: int = 0
            self.sma_start_iter: int = self.args.sma_start_iter
            self.logger.info(f"SMA will start at: {self.sma_start_iter}")

        # 4-1. Correlation
        if self.defaults.loss_type == 'multiclass':
            J: int = NetworkInitializer._out_features[self.args.data][0]  # number of outcome classes
            K: int = len(self.train_domains)                              # number of training domains
            _rho = torch.randn(K, J + 1, device=self.device, requires_grad=True)
            self._rho = nn.Parameter(_rho)
        else:
            K: int = len(self.train_domains)
            _rho = torch.zeros(K, device=self.device, requires_grad=True)
            self._rho = nn.Parameter(data=_rho)

        # 4-2. Sigma (for regression only)
        if self.defaults.loss_type == 'regression':
            self.sigma = nn.Parameter(torch.ones(1, device=self.device), requires_grad=True)

        # 5. temperature
        self.temperature: typing.Union[torch.FloatTensor, float] = 1.0

    def _init_transforms(self) -> None:
        """Initialize transforms."""

        transform_constructor: object = InputTransforms.get(self.args.data)
        self.eval_transform = transform_constructor(augmentation=False)
        self.selection_transform = transform_constructor(
            augmentation=self.args.augmentation_selection, randaugment=self.args.randaugment_selection,
        )
        self.outcome_transform = transform_constructor(
            augmentation=self.args.augmentation_outcome, randaugment=self.args.randaugment_outcome,
        )

        self.logger.info(f"Selection transform: {self.selection_transform}")
        self.logger.info(f"Outcome transform: {self.outcome_transform}")
        self.logger.info(f"Eval transform: {self.eval_transform}")

    def _init_cuda(self) -> None:

        # modules
        self.outcome_encoder.to(self.device)
        self.outcome_head.to(self.device)
        self.selection_encoder.to(self.device)
        self.selection_head.to(self.device)

        # transforms
        self.selection_transform.to(self.device)
        self.outcome_transform.to(self.device)
        self.eval_transform.to(self.device)

        self.logger.info(f"GPU configuration: {self.device}")

    def _init_optimization(self) -> None:

        params: typing.List[dict] = [
            {'params': self.selection_encoder.parameters()},
            {'params': self.selection_head.parameters()},
            {'params': self.outcome_encoder.parameters()},
            {'params': self.outcome_head.parameters()},
        ]

        # optimizer
        self.optimizer = configure_optimizer(
            params=params,
            name=self.args.optimizer,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.logger.info(f"Optimizer (name={self.optimizer.__class__.__name__})")

        # scheduler
        if self.args.scheduler is not None:
            self.scheduler = configure_learning_rate_scheduler(
                optimizer=self.optimizer,
                name=self.args.scheduler,
                epochs=self.args.epochs,
                warmup_epochs=self.args.scheduler_lr_warmup,
                min_lr=self.args.scheduler_min_lr,
            )
        else:
            self.scheduler = None
        self.logger.info(f"LR Scheduler (name={self.scheduler.__class__.__name__})")

        # correlation parameters
        _corr_params: list = [self._rho, ]
        if self.defaults.loss_type == 'regression':
            _corr_params += [self.sigma, ]

        # correlation optimizer
        self.corr_optimizer = configure_optimizer(
            params=_corr_params,
            name=self.args.corr_optimizer,
            lr=self.args.corr_lr,
            weight_decay=self.args.corr_weight_decay,
        )
        self.logger.info(f"Correlation optimizer (name={self.corr_optimizer.__class__.__name__}).")

        # correlation scheduler
        if self.args.corr_scheduler is not None:
            self.corr_scheduler = configure_learning_rate_scheduler(
                optimizer=self.corr_optimizer,
                name=self.args.corr_scheduler,
                epochs=self.args.epochs,
                warmup_epochs=self.args.corr_scheduler_lr_warmup,
                min_lr=self.args.corr_scheduler_min_lr,
            )
        else:
            self.corr_scheduler = None
        self.logger.info(f"Correlation LR scheduler (name={self.corr_scheduler.__class__.__name__}).")

    def _init_checkpoint_dir(self) -> None:
        """..."""

        _ckpt_base: str = f"./checkpoints/HeckmanDG/{self.args.data}/{self.args.backbone}/"
        if self.args.data in ('poverty', 'povertymap', ):
            self.checkpoint_dir = os.path.join(_ckpt_base, f"{self.defaults.fold}/{self.args.hash}")
        else:
            self.checkpoint_dir = os.path.join(_ckpt_base, f"{self.args.hash}")
        os.makedirs(self.checkpoint_dir, exist_ok=False)

        # save command-line arguments to a yaml file under the checkpoint directory
        _config_file = os.path.join(self.checkpoint_dir, 'configuration.yaml')
        with open(_config_file, 'w') as fp:
            yaml.dump(vars(self.args), fp, default_flow_style=False)

    def fit(self, dataset: MultipleDomainCollection) -> typing.Dict[str, MetricAccumulator]:

        # Get datasets
        train_sets: typing.List[torch.utils.data.Dataset] = dataset.get_train_data(as_dict=False)
        id_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_id_validation_data(as_dict=False)
        ood_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_ood_validation_data(as_dict=False)
        test_sets: typing.List[torch.utils.data.Dataset] = dataset.get_test_data(as_dict=False)

        # Configure train sampler; returns `None` if neither of the following options are True
        #   - `uniform_across_domains`: balanced with respect to domain indicators
        #   - `uniform_across_targets`: balanced with respect to class labels
        train_sampler = configure_train_sampler(
            datasets=train_sets,
            use_domain=self.args.uniform_across_domains,
            use_target=self.args.uniform_across_targets,
        )

        _fit_kwargs: dict = {
            'train_set': ConcatDataset(train_sets),
            'id_validation_set': ConcatDataset(id_validation_sets) if id_validation_sets is not None else None,
            'ood_validation_set': ConcatDataset(ood_validation_sets) if ood_validation_sets is not None else None,
            'test_set': ConcatDataset(test_sets),
            'train_sampler': train_sampler,
        }

        return self._fit(**_fit_kwargs)

    def _fit(self,
             train_set: torch.utils.data.Dataset,
             id_validation_set: torch.utils.data.Dataset,
             ood_validation_set: torch.utils.data.Dataset,
             test_set: torch.utils.data.Dataset,
             train_sampler: torch.utils.data.Sampler = None,
             **kwargs, ) -> typing.Dict[str, MetricAccumulator]:

        # Instantiate train loader
        train_loader_configs = dict(
            batch_size=self.args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.args.prefetch_factor,
            num_workers=self.args.num_workers,
        )
        train_loader = DataLoader(train_set, **train_loader_configs)

        # Instantiate {id_val, ood_val, test} loaders
        eval_loader_configs = dict(
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=self.args.eval_prefetch_factor,
            num_workers=self.args.eval_num_workers,
        )

        _id_val: bool = False
        if id_validation_set is not None:
            id_val_loader = DataLoader(id_validation_set, **eval_loader_configs)    # ID
            _id_val = True

        _ood_val: bool = False
        if ood_validation_set is not None:
            ood_val_loader = DataLoader(ood_validation_set, **eval_loader_configs)  # OOD
            _ood_val = True

        test_loader = DataLoader(test_set, **eval_loader_configs)  # Test

        # Buffers to store metric values of every epoch
        train_metrics = MetricAccumulator(name='train')
        id_val_metrics = MetricAccumulator(name='id_val') if _id_val else None
        ood_val_metrics = MetricAccumulator(name='ood_val') if _ood_val else None
        test_metrics = MetricAccumulator(name='test')
        selection_train_metrics = MetricAccumulator(name='selection/train')
        selection_id_val_metrics = MetricAccumulator(name='selection/id_val')

        # Model selector
        if (not _id_val) and (self.defaults.model_selection == 'id'):
            raise ValueError(f"The id-validation set not provided. Model selection incompatible.")
        if (not _ood_val) and (self.defaults.model_selection == 'ood'):
            raise ValueError(f"The ood-validation set not provided. Model selection incompatible.")
        model_selector = ModelSelector(
            metric=self.defaults.model_selection_metric,
            patience=self.args.early_stopping,
        )
        self.logger.info(
            f"Choosing best model based on {model_selector.metric}. "
            f"Validation data = {self.defaults.model_selection}. "
            f"Early stopping = {model_selector.patience}. "
        )

        # calibrate selection model predictions
        if self.args.calibrate:
            self.temperature = self._calibrate(data_loader=id_val_loader)
            self.temperature.requires_grad = False

        _msg_fmt: str = f">{len(str(self.args.epochs))}"
        for epoch in range(1, self.args.epochs + 1):

            # Check termination condition
            if model_selector.terminate:
                self.logger.info(f"Early stopping at epoch {epoch:,}/{self.args.epochs}.")
                break

            # Train (outcome)
            epoch_history: dict = self.train(train_loader=train_loader)
            train_metrics.update(epoch_history)
            msg: str = f"(Train) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Log (correlation)
            if self.defaults.loss_type != 'multiclass':  # TODO: implement for `multiclass`
                self._log_correlation_info(epoch=epoch, total_epochs=self.args.epochs)

            # Evaluation (selection / train)
            selection_train_epoch_history: dict = self.evaluate_selection_model(train_loader, suffix='Train')
            selection_train_metrics.update(selection_train_epoch_history)
            msg: str = f"(Selection/Train) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in selection_train_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Evaluation (selection / id_val)
            selection_id_val_epoch_history: dict = self.evaluate_selection_model(id_val_loader, suffix='ID')
            selection_id_val_metrics.update(selection_id_val_epoch_history)
            msg: str = f"(Selection/   ID) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in selection_id_val_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Evaluation (outcome / ID)
            if _id_val:
                id_val_epoch_history: dict = self.evaluate(id_val_loader, suffix='ID')
                id_val_metrics.update(id_val_epoch_history)
                msg: str = f"(   ID) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
                msg += " | ".join([f"{k}: {v:.4f}" for k, v in id_val_metrics[-1].items()])
                self.logger.info(msg, extra={'markup': True})

            # Evaluation (outcome / OOD)
            if _ood_val:
                ood_val_epoch_history: dict = self.evaluate(ood_val_loader, suffix='OOD')
                ood_val_metrics.update(ood_val_epoch_history)
                msg: str = f"(  OOD) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
                msg += " | ".join([f"{k}: {v:.4f}" for k, v in ood_val_metrics[-1].items()])
                self.logger.info(msg, extra={'markup': True})

            # Evaluation (outcome / Test)
            test_epoch_history = self.evaluate(test_loader, suffix='Test')
            test_metrics.update(test_epoch_history)
            msg: str = f"( Test) Epoch [{epoch:{_msg_fmt}}/{self.args.epochs}]: "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics[-1].items()])
            self.logger.info(msg, extra={'markup': True})

            # Update model selection status
            if self.defaults.model_selection == 'id':
                if not _id_val:
                    raise NotImplementedError
                value: torch.Tensor = id_val_epoch_history[model_selector.metric]
            elif self.defaults.model_selection == 'ood':
                if not _ood_val:
                    raise NotImplementedError
                value: torch.Tensor = ood_val_epoch_history[model_selector.metric]
            else:
                raise ValueError(f"Not supported: {self.defaults.model_selection}. ")
            model_selector.update(
                value=value,
                model={
                    'outcome_encoder': self.outcome_encoder,
                    'outcome_head': self.outcome_head,
                },
                step=epoch,
                logger=self.logger,
            )

            # Logging with https://wandb.ai
            log = dict()
            log.update(train_metrics[-1])
            log.update(id_val_metrics[-1] if _id_val else {})
            log.update(ood_val_metrics[-1] if _ood_val else {})
            log.update(test_metrics[-1])
            log.update(selection_train_metrics[-1])
            log.update(selection_id_val_metrics[-1] if _id_val else {})
            log.update(
                {
                    f"best/val_{model_selector.metric}": model_selector.best_value,
                    f"best/test_{model_selector.metric}": test_metrics.get(
                        index=model_selector.best_step - 1,
                        prefix=False,
                    )[model_selector.metric]
                }
            )
            log.update(
                {
                    'misc/lr': self._get_current_lr(),
                    'misc/best_epoch': model_selector.best_step,
                    'misc/no_improvement': model_selector.no_improvement,
                }
            )
            # wandb.log(log, step=epoch)

            # Save current best model (when it has improved)
            if model_selector.no_improvement == 0:
                ckpt: str = os.path.join(self.checkpoint_dir, f'ckpt.best.pth.tar')
                self.logger.info(f"Saving best checkpoint to {ckpt}")
                self.save_checkpoint(ckpt, epoch, which='both', history=log)

            # Periodically save model
            if epoch % self.args.save_every == 0:
                _save_fmt: str = f"0{len(str(self.args.epochs))}"
                ckpt: str = os.path.join(self.checkpoint_dir, f'ckpt.{epoch:{_save_fmt}}.pth.tar')
                self.logger.info(f"Saving intermediate checkpoint to: {ckpt}")
                self.save_checkpoint(ckpt, epoch=epoch, which='both', history=log)

        self.logger.info("Finished model training.")

        # Logging for best model. Note that as we start our index from 1,
        #   we should subtract 1 when retrieving the best metric values.
        best_epoch: int = model_selector.best_step
        self.logger.info(f"Best model at epoch {best_epoch:,}, metric={model_selector.metric}")
        msg: str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics[best_epoch - 1].items()])
        self.logger.info("[white]" + msg, extra={'markup': True})

        if _id_val:  # ID validation
            msg: str = " | ".join([f"{k}: {v:.4f}" for k, v in id_val_metrics[best_epoch - 1].items()])
            self.logger.info("[blue]" + msg, extra={'markup': True})

        if _ood_val: # OOD validation
            msg: str = " | ".join([f"{k}: {v:.4f}" for k, v in ood_val_metrics[best_epoch - 1].items()])
            self.logger.info("[magenta]" + msg, extra={'markup': True})

        # Test evaluation
        msg = " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics[best_epoch - 1].items()])
        self.logger.info("[red]" + msg, extra={'markup': True})

        # Write metrics to csv file.
        selection_train_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'selection_train_metrics.csv'))
        selection_id_val_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'selection_id_val_metrics.csv'))
        train_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'train_metrics.csv'))
        if _id_val:
            id_val_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'id_val_metrics.csv'))
        if _ood_val:
            ood_val_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'ood_val_metrics.csv'))
        test_metrics.write_to_csv(os.path.join(self.checkpoint_dir, 'test_metrics.csv'))

        return {
            'selection/train': selection_train_metrics,
            'selection/id_val': selection_id_val_metrics,
            'outcome/train': train_metrics,
            'outcome/id_val': id_val_metrics if _id_val else {},
            'outcome/ood_val': ood_val_metrics if _ood_val else {},
            'outcome/test': test_metrics,
        }

    def train(self, train_loader: torch.utils.data.DataLoader) -> typing.Dict[str, torch.FloatTensor]:

        self._set_learning_phase(train=True)

        # buffers
        losses = torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
        losses_selected = torch.zeros_like(losses)      # FIXME: temporary, remove later
        losses_not_selected = torch.zeros_like(losses)  # FIXME: temporary, remove later
        losses_g = torch.zeros_like(losses)             # FIXME: temporary, remove later
        y_pred, y_true = list(), list()
        s_pred, s_true = list(), list()
        eval_group = list()
        metrics = dict()

        # train (iterate over data loader for a single epoch)
        _pbar_kwargs = dict(
            total=len(train_loader),
            transient=True,
            description=f"HeckmanDG training (data={self.args.data}, backbone={self.args.backbone}, #.iter={len(train_loader):,})"
        )
        for i, batch in track(enumerate(train_loader), **_pbar_kwargs):

            # fetch data
            x = batch['x'].to(self.device, non_blocking=True)
            target = batch['y'].to(self.device, non_blocking=True)
            domain = batch['domain'].to(self.device, non_blocking=True)
            s_true_2d = domain.view(-1, 1).eq(
                torch.tensor(self.train_domains, dtype=torch.long, device=self.device).view(1, -1)
            ).long()                                         # (B, K)
            s_true_1d = s_true_2d.nonzero(as_tuple=True)[1]  # (B,  )

            # transform inputs
            x_sel = self.selection_transform(x)
            x_out = self.outcome_transform(x)

            # forward
            probits_or_resp = self.outcome_head(self.outcome_encoder(x_out))
            if self.defaults.loss_type != 'multiclass':
                probits_or_resp = probits_or_resp.squeeze()
                assert probits_or_resp.shape == target.shape, "(B,  )"
            s_pred_in_probits = self.selection_head(self.selection_encoder(x_sel))
            s_pred_in_probits.div_(self.temperature)  # no effect if not `self.args.calibrate`

            rho = torch.tanh(self._rho[s_true_1d])    # get $\rho$ for each example in current mini-batch

            # compute loss
            if self.defaults.loss_type == 'regression':
                loss, loss_sel, loss_not_sel = self._cross_domain_regression_loss(  # FIXME: 
                    y_pred=probits_or_resp,
                    y_true=target,
                    s_pred=s_pred_in_probits,
                    s_true=s_true_1d,
                    rho=rho,
                    sigma=self.sigma,
                )
            elif self.defaults.loss_type == 'binary':
                loss, loss_sel, loss_not_sel = self._cross_domain_binary_classification_loss(  # FIXME:
                    y_pred=probits_or_resp,
                    y_true=target,
                    s_pred=s_pred_in_probits,
                    s_true=s_true_1d,
                    rho=rho,
                )
            elif self.defaults.loss_type == 'multiclass':
                loss, loss_sel, loss_not_sel = self._cross_domain_multiclass_classification_loss(  # FIXME: 
                    y_pred=probits_or_resp,
                    y_true=target,
                    s_pred=s_pred_in_probits,
                    s_true=s_true_1d,
                    rho=rho,
                    approximate=True,
                )
            else:
                raise ValueError

            # backward & update
            self.optimizer.zero_grad()
            self.corr_optimizer.zero_grad()
            loss.backward()
            _ = self.optimizer.step()
            _ = self.corr_optimizer.step()

            # accumulate loss value for logging
            with torch.no_grad():
                losses[i] = loss.detach().clone()
                losses_selected[i] = loss_sel.detach().clone()
                losses_not_selected[i] = loss_not_sel.detach().clone()
                losses_g[i] = self.selection_loss(s_pred=s_pred_in_probits,
                                                  s_true=s_true_1d).detach().clone()

            # accumulate {y_pred, y_true, s_pred, s_true} for evaluation
            with torch.no_grad():
                y_pred += [ probits_or_resp.detach().clone() ]
                y_true += [ target ]
                s_pred += [ s_pred_in_probits.detach().clone() ]
                s_true += [ s_true_1d ]
                eval_group += [ batch['eval_group'].to(self.device) ]

        # Update learning rate (after full epoch)
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler.step()
        if isinstance(self.corr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.corr_scheduler.step()

        # Concatenate lists to tensors
        with torch.no_grad():
            y_pred, y_true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
            s_pred, s_true = torch.cat(s_pred, dim=0), torch.cat(s_true, dim=0)
            eval_group = torch.cat(eval_group, dim=0)

        # Compute metrics (after a complete sweep through the whole dataset)
        with torch.no_grad():
            metrics['loss'] = losses.mean()
            metrics['loss_selected'] = losses_selected.mean()
            metrics['loss_not_selected'] = losses_not_selected.mean()
            metrics['loss_g'] = losses_g.mean()
            evaluator = MetricEvaluator(data=self.args.data, exclude_metrics=['loss', ])
            _metrics: dict = evaluator.evaluate(
                y_pred=self._process_pred_for_eval(y_pred),
                y_true=y_true,
                group=eval_group,
            )
            metrics.update(_metrics)

        return metrics  # dictionary

    @torch.no_grad()
    def evaluate_selection_model(self,
                                 data_loader: torch.utils.data.DataLoader,
                                 suffix: str = ' ') -> typing.Dict[str, torch.FloatTensor]:

        # eval mode
        self.selection_encoder.eval()
        self.selection_head.eval()

        s_pred, s_true = list(), list()
        metrics = dict()

        _pbar_kwargs = dict(
            total=len(data_loader),
            transient=True,
            description=f":roller_coaster: (Selection) evaluation (data={self.args.data}, backbone={self.args.backbone}, split={suffix}) "
        )
        for _, batch in track(enumerate(data_loader), **_pbar_kwargs):

            x = batch['x'].to(self.device, non_blocking=True)
            if self.eval_transform is not None:
                x = self.eval_transform(x)

            # Shape; (B, K)
            s_pred_in_probits = self.selection_head(self.selection_encoder(x))
            s_pred += [s_pred_in_probits]
            s_true += [batch['domain'].to(self.device)]

        s_pred = torch.cat(s_pred, dim=0)  # (B, K); still in probits
        s_pred.div_(self.temperature)      # (B, K); still in probits but scaled with temperature
        s_pred = self._normal.cdf(s_pred)  # (B, K); now in probabilities
        s_true = torch.cat(s_true, dim=0)  # (B,  ); original domain indicators (may not be 0,1,...,K-1)
        s_true = s_true.view(-1, 1).eq(
            torch.tensor(self.train_domains, dtype=torch.long, device=self.device).view(1, -1)
        ).long()                           # (B, K) <- (B, 1) * (1, K)

        metrics['accuracy'] = multilabel_accuracy(preds=s_pred, targets=s_true, average='macro')
        metrics['recall'] = multilabel_recall(preds=s_pred, targets=s_true, average='macro')
        metrics['precision'] = multilabel_precision(preds=s_pred, targets=s_true, average='macro')
        metrics['f1'] = multilabel_f1(preds=s_pred, targets=s_true, average='macro')

        return metrics  # dictionary

    @torch.no_grad()
    def evaluate_outcome_model(self,
                               data_loader: torch.utils.data.DataLoader,
                               suffix: str = ' ',
                               ) -> typing.Dict[str, torch.FloatTensor]:
        return self.evaluate(data_loader=data_loader, suffix=suffix)

    @torch.no_grad()
    def evaluate(self,
                 data_loader: torch.utils.data.DataLoader,
                 suffix: str = ' ') -> typing.Dict[str, torch.FloatTensor]:

        # eval mode
        self.outcome_encoder.eval()
        self.outcome_head.eval()

        y_pred, y_true = list(), list()
        eval_group = list()
        metrics = dict()

        _desc: str = f":rocket: (Outcome) evaluation (data={self.args.data}, backbone={self.args.backbone}, split={suffix}) "
        for _, batch in track(enumerate(data_loader), total=len(data_loader), transient=True, description=_desc):

            y = batch['y'].to(self.device, non_blocking=True)  # (B,  ) always
            x = batch['x'].to(self.device, non_blocking=True)  # (B, *)
            if self.eval_transform is not None:
                x = self.eval_transform(x)

            # Shape; (B, ) for regression or binary, (B, J) for multiclass
            probits_or_resp = self.outcome_head(self.outcome_encoder(x))
            if self.defaults.loss_type != 'multiclass':
                probits_or_resp = probits_or_resp.squeeze()  # (B,  ) <- (B, 1)
            y_pred += [probits_or_resp]
            y_true += [y]
            eval_group += [batch['eval_group'].to(self.device)]

        y_pred, y_true = torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0)
        eval_group = torch.cat(eval_group, dim=0)

        evaluator = MetricEvaluator(data=self.args.data)
        _metrics: dict = evaluator.evaluate(
            y_pred=self._process_pred_for_eval(y_pred),  # probits to probabilities
            y_true=y_true,
            group=eval_group,
        )
        metrics.update(_metrics)

        return metrics  # dictionary

    @torch.no_grad()
    def _process_pred_for_eval(self, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        """Add function docstring."""
        if self.defaults.loss_type == 'regression':
            return y_pred
        elif self.defaults.loss_type == 'binary':
            return self._normal.cdf(y_pred)
        elif self.defaults.loss_type == 'multiclass':
            return F.softmax(y_pred, dim=1)
        else:
            raise ValueError(f"`{self.defaults.loss_type}` unrecognized.")

    def selection_loss(self, s_pred: torch.FloatTensor,
                             s_true: torch.LongTensor, ) -> torch.FloatTensor:
        """Loss for training domain selection model(s)."""
        
        s_pred_in_probs = self._normal.cdf(s_pred)
        s_true_2d = F.one_hot(s_true, num_classes=s_pred.shape[1]).float()
        return F.binary_cross_entropy(s_pred_in_probs, s_true_2d)


    def _cross_domain_regression_loss(self,
                                      y_pred: torch.FloatTensor,
                                      y_true: torch.FloatTensor,
                                      s_pred: torch.FloatTensor,
                                      s_true: torch.LongTensor,
                                      rho   : torch.FloatTensor,
                                      sigma : torch.FloatTensor, ) -> torch.FloatTensor:
        """
        Arguments:
            y_pred : 1d `torch.FloatTensor` of shape (N,  ),
            y_true : 1d `torch.FloatTensor` of shape (N,  ),
            s_pred : 2d `torch.FloatTensor` of shape (N, K),
            s_true : 1d `torch.LongTensor`  of shape (N,  ),
            rho    : 1d `torch.FloatTensor` of shape (N,  ),
            sigma  : 1d `torch.FloatTensor` of shape (1,  ), singleton.
        """

        if (y_pred.ndim == 2) and (y_pred.size(1) == 1):
            y_pred = y_pred.squeeze(1)
        if (y_true.ndim == 2) and (y_true.size(1) == 1):
            y_true = y_true.squeeze(1)

        _epsilon: float = 1e-7
        _normal = torch.distributions.Normal(loc=0., scale=1.)

        # Gather values from `s_pred` those that correspond to the true domains
        s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).squeeze()

        # -\log p[S_k = 1, y] = -\log p(y) -\log p(S_k = 1 | y) ; shape = (N,  )
        loss_selected = - torch.log(
            _normal.cdf(
                (s_pred_k + rho * (y_true - y_pred).div(_epsilon + sigma)) / (_epsilon + torch.sqrt(1 - rho ** 2))
            ) + _epsilon
        ) + 0.5 * (
            torch.log(2 * torch.pi * (sigma ** 2)) \
                + F.mse_loss(y_pred, y_true, reduction='none').div(_epsilon + sigma ** 2)
        )

        if (self.args.freeze_selection_encoder & self.args.freeze_selection_head):
            return loss_selected.mean()

        # domain indicators in 2d; (N, K) <- (N,  )
        s_true_2d = torch.zeros_like(s_pred).scatter_(
            dim=1,
            index=s_true.view(-1, 1),
            src=torch.ones_like(s_pred),
        )

        # -\log Pr[S_l = 0] for l \neq k
        #   shape; (N(K-1),  )
        loss_not_selected = - torch.log(1 - _normal.cdf(s_pred) + _epsilon)                    # (N     , K)
        loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.)  # (N     , K)
        loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())            # (N(K-1),  )

        return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()

    def _cross_domain_binary_classification_loss(self,
                                                 y_pred: torch.FloatTensor,
                                                 y_true: torch.LongTensor,
                                                 s_pred: torch.FloatTensor,
                                                 s_true: torch.LongTensor,
                                                 rho   : torch.FloatTensor, ) -> torch.FloatTensor:
        """
        Arguments:
            y_pred : 1d `torch.FloatTensor` of shape (N,  ); in probits.
            y_true : 1d `torch.LongTensor`  of shape (N,  ); with values in {0, 1}.
            s_pred : 2d `torch.FloatTensor` of shape (B, K); in probits.
            s_true : 1d `torch.LongTensor`  of shape (N,  ); with values in [0, K-1].
            rho    : 1d `torch.FloatTensor` of shape (N,  ); with values in [-1, 1].
        Returns:
            ...
        """

        _epsilon: float = 1e-7
        _normal = torch.distributions.Normal(loc=0., scale=1.)

        # Gather from `s_pred` values with indices that correspond to the true domains
        s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).squeeze()  # (N,  )

        # - log Pr[S_k = 1, Y = 1]; shape = (N,  )
        loss_selected_pos = - y_true.float() * torch.log(
            self._bivariate_normal_cdf(a=s_pred_k, b=y_pred, rho=rho) + _epsilon,
        )
        loss_selected_pos = torch.nan_to_num(loss_selected_pos, nan=0., posinf=0., neginf=0.)
        loss_selected_pos = loss_selected_pos[y_true.bool()]

        # - log Pr[S_k = 1, Y = 0]; shape = (N,  )
        loss_selected_neg = - (1 - y_true.float()) * torch.log(
            _normal.cdf(s_pred_k) - self._bivariate_normal_cdf(a=s_pred_k, b=y_pred, rho=rho) + _epsilon
        )
        loss_selected_neg = torch.nan_to_num(loss_selected_neg, nan=0., posinf=0., neginf=0.)
        loss_selected_neg = loss_selected_neg[(1 - y_true).bool()]

        loss_selected = torch.cat([loss_selected_pos, loss_selected_neg], dim=0)

        # Create a 2d indicator for `s_true`
        #   Shape; (N, K)
        s_true_2d = torch.zeros_like(s_pred).scatter_(
            dim=1, index=s_true.view(-1, 1), src=torch.ones_like(s_pred)
        )

        # -\log Pr[S_l = 0] for l \neq k
        loss_not_selected = - torch.log(1 - _normal.cdf(s_pred) + _epsilon)     # (N, K)
        loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.)  # (N, K)
        loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())            # (NK - N,  )
        
        return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()

        # total_count: int = loss_selected.numel() + loss_not_selected.numel()
        # sel_weight: float = total_count / len(loss_selected)
        # not_sel_weight: float = total_count / len(loss_not_selected)
        # return torch.cat([loss_selected * sel_weight, loss_not_selected * not_sel_weight], dim=0).mean()

    @classmethod
    def _bivariate_normal_cdf(cls, a: torch.FloatTensor, b: torch.FloatTensor, rho: torch.FloatTensor, steps: int = 100):
        """
        Approximation of standard bivariate normal cdf using the trapezoid rule.
        The decomposition is based on:
            Drezner, Z., & Wesolowsky, G. O. (1990).
            On the computation of the bivariate normal integral.
            Journal of Statistical Computation and Simulation, 35(1-2), 101-107.
        Arguments:
            a: 1d `torch.FloatTensor` of shape (N,  )
            b: 1d `torch.FloatTensor` of shape (N,  )
            rho: 1d `torch.FloatTensor` of shape (N,  )
        Returns:
            1d `torch.FloatTensor` of shape (N,  )
        """

        _normal = torch.distributions.Normal(loc=0., scale=1.)
        a, b = a.view(-1, 1), b.view(-1, 1)  # for proper broadcasting with x

        grids: typing.List[torch.FloatTensor] = [
            cls._linspace_with_grads(start=0, stop=r, steps=steps, device=a.device) for r in rho
        ]                              #  N * (steps,  )
        x = torch.stack(grids, dim=0)  # (N, steps)
        y = 1 / torch.sqrt(1 - torch.pow(x, 2)) * torch.exp(
            - (torch.pow(a, 2) + torch.pow(b, 2) - 2 * a * b * x) / (2 * (1 - torch.pow(x, 2)))
        )

        return _normal.cdf(a.squeeze()) * _normal.cdf(b.squeeze()) + \
            (1 / (2 * torch.pi)) * torch.trapezoid(y=y, x=x)

    @classmethod
    def _linspace_with_grads(cls, start: torch.FloatTensor, stop: torch.FloatTensor, steps: int, device: str = 'cpu'):
        """
        Creates a 1d grid while preserving gradients associated with `start` and `stop`.
        Reference:
            https://github.com/esa/torchquad/blob/4be241e8462949abcc8f1ace48d7f8e5ee7dc136/torchquad/integration/utils.py#L7
        """
        grid = torch.linspace(0, 1, steps, device=device)  # create 0 ~ 1 equally spaced grid
        grid *= stop - start                               # scale grid to desired range
        grid += start
        return grid

    def _cross_domain_multiclass_classification_loss(self,
                                                     y_pred: torch.FloatTensor,
                                                     y_true: torch.LongTensor,
                                                     s_pred: torch.FloatTensor,
                                                     s_true: torch.LongTensor,
                                                     rho: torch.FloatTensor,       # shape; (N, J+1)
                                                     approximate: bool = False,    # logistic approx.
                                                     **kwargs, ) -> torch.FloatTensor:
        """Multinomial loss with logistic approximation available."""

        _eps: float = 1e-7
        _normal = torch.distributions.Normal(loc=0., scale=1.)
        _float_type = y_pred.dtype  # creating new tensors (in case of amp)

        B = int(y_true.size(0))  # batch size
        J = int(y_pred.size(1))  # number of classes
        K = int(s_pred.size(1))  # number of (training) domains

        s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).flatten()
        assert len(s_pred_k) == len(s_pred)

        # matrix of y(probit) differences (with respect to its true outcome)
        y_pred_diff = torch.zeros(B, J-1, dtype=_float_type, device=self.device)
        for j in range(J):
            col_mask = torch.arange(0, J, device=self.device).not_equal(j)
            row_mask = y_true.eq(j)
            y_pred_diff[row_mask, :] = (
                y_pred[:, j].view(-1, 1) - y_pred[:, col_mask]
            )[row_mask, :]

        assert len(rho) == len(y_pred)
        assert rho.shape[1] == (y_pred.shape[1] + 1)

        C_tilde_list = list()
        for i in range(B):
            L = torch.eye(J + 1, device=rho.device)  # construct a lower triangular matrix
            L[-1] = rho[i]                           # fill in params
            C = MatrixOps.cov_to_corr(L @ L.T)       # (J+1, J+1)
            j: int = y_true[i].item()                # true target index
            Cy = C[:J, :J].clone()                   # (J, J)
            Cy_diff = MatrixOps.compute_cov_of_error_differences(Cy, j=j)  # (J-1, J-1)
            C_tilde = torch.empty(J, J, device=rho.device)                 # (J, J)
            C_tilde[:J-1, :J-1] = Cy_diff
            not_j = torch.arange(0, J, device=rho.device).not_equal(j)
            not_j = not_j.nonzero(as_tuple=True)[0]
            C_tilde[-1, :-1] = C[-1, j] - C[-1, not_j]                     # (1,  ) - (1, J-1)
            C_tilde[:-1, -1] = C[j, -1] - C[not_j, -1]                     # ...
            C_tilde[-1, -1] = C[-1, -1]                                    # equals 1
            C_tilde = MatrixOps.make_positive_definite(C_tilde)            # not necessary
            C_tilde_list += [C_tilde]

        # Cholesky decomposition; (B, J, J)
        L = torch.linalg.cholesky(torch.stack(C_tilde_list, dim=0))
        L_lower = L - torch.diag_embed(
            torch.diagonal(L, offset=0, dim1=1, dim2=2),
            dim1=1, dim2=2
        )

        def truncnorm_rvs_recursive(loc, scale, lower_clip, max_iter: int = 10):
            """Add function docstring."""
            n: int = len(loc)
            q = np.random.normal(loc, scale, size=(n, ))
            mask = (q < lower_clip)  # True if not valid sample of truncated normal
            if np.any(mask):
                if max_iter > 0:
                    # recursively sample
                    q[mask] = truncnorm_rvs_recursive(
                        loc=loc[mask],
                        scale=scale[mask],
                        lower_clip=lower_clip[mask],
                        max_iter=max_iter-1,
                    )
                else:
                    q[mask] = lower_clip[mask] + 1e-5
            return q

        # GHK algorithm
        _probs = torch.ones(B, J, dtype=_float_type, device=self.device)
        v = torch.zeros_like(_probs)
        for l in range(J):         # 0, 1, ..., J-2, J-1
            if l < (J - 1):
                lower_trunc = - (  # (B,  )
                    y_pred_diff[:, l] - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l]
                ).div(torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l])                    # (B,  )
            else:
                lower_trunc = - (  # (B,  )
                    s_pred_k - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l]  # (B,  )
                ).div(torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l])                    # (B,  )

            # TODO: by far this is the fastest implementation, but can we do better?
            # sample from truncated normal (in batches)
            lower_trunc_numpy = lower_trunc.detach().cpu().numpy()
            samples = truncnorm_rvs_recursive(
                loc=np.zeros(B), scale=np.ones(B),
                lower_clip=lower_trunc_numpy, max_iter=5,
            )
            v[:, l] = torch.from_numpy(samples).to(self.device).flatten()
            _probs[:, l] = 1. - _normal.cdf(lower_trunc)

        if approximate:
            y_probs = F.softmax(y_pred, dim=1).gather(dim=1, index=y_true.view(-1, 1)).flatten()
        else:
            y_probs = torch.prod(_probs[:, :-1], dim=1).flatten()

        # Pr[Y = j, S_k = 1 | X]
        y_s_joint_probs = y_probs * _probs[:, -1].flatten()  # the last column of `_probs` gives Pr[S_k = 1 | Y = j, X]
        loss_selected = - torch.log(y_s_joint_probs + _eps)

        # Pr[S_l = 0 | X], for l \neq k
        s_true_2d = torch.zeros_like(s_pred).scatter_(
            dim=1, index=s_true.view(-1, 1), src=torch.ones_like(s_pred),
        )
        loss_not_selected = - torch.log(1. - _normal.cdf(s_pred) + _eps)
        loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.,)
        loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())  # s_true_2d = 0

        return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()

    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    @torch.no_grad()
    def _update_sma(self) -> None:
        """Update simple moving average model (used at evaluation stage.)"""
        _new_encoder_state_dict = dict()
        _new_head_state_dict = dict()
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            _weight = float(self.sma_count) / float(1. + self.sma_count)
            for (name, p), (_, p_sma) in zip(self.outcome_encoder.state_dict().items(), self.outcome_encoder_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    _new_encoder_state_dict[name] = \
                        p_sma.data.detach().clone() * _weight + p.data.detach().clone() * (1. - _weight)
            for (name, p), (_, p_sma) in zip(self.outcome_head.state_dict().items(), self.outcome_head_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    _new_head_state_dict[name] = \
                        p_sma.data.detach().clone() * _weight + p.data.detach().clone() * (1. - _weight)
        else:
            for (name, p), (_, p_sma) in zip(self.outcome_encoder.state_dict().items(), self.outcome_encoder_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    _new_encoder_state_dict[name] = p.data.detach().clone()
            for (name, p), (_, p_sma) in zip(self.outcome_head.state_dict().items(), self.outcome_head_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    _new_head_state_dict[name] = p.data.detach().clone()

        # override state dict
        self.outcome_encoder_sma.load_state_dict(_new_encoder_state_dict)
        self.outcome_head_sma.load_state_dict(_new_head_state_dict)

    def save_checkpoint(self, path: str, epoch: int, which: str = 'both', **kwargs) -> None:
        """Save model to a `***.pth.tar` file."""
        ckpt = {
            'selection_encoder': self.selection_encoder.state_dict(),
            'selection_head': self.selection_head.state_dict(),
            'outcome_encoder': self.outcome_encoder.state_dict(),
            'outcome_head': self.outcome_head.state_dict(),
        }
        if which == 'both':
            pass
        elif which == 'selection':
            ckpt = {k: v for k, v in ckpt.items() if k.startswith('selection')}
        elif which == 'outcome':
            ckpt = {k: v for k, v in ckpt.items() if k.startswith('outcome')}
        else:
            raise RuntimeError

        ckpt['optimizer'] = self.optimizer.state_dict()
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            ckpt['scheduler'] = self.scheduler.state_dict()
        else:
            ckpt['scheduler'] = None
        ckpt['epoch'] = epoch

        # correlation
        ckpt['rho'] = self._rho.clone().detach().cpu()
        if self.defaults.loss_type == 'regression':
            ckpt['sigma'] = self.sigma.clone().detach().cpu()

        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_selection_weights_from_checkpoint(self,
                                               path: str,
                                               encoder_keys: typing.Iterable[str] = ['encoder', 'selection_encoder'],
                                               head_keys: typing.Iterable[str] = ['selection_head', ],
                                               ) -> None:
        
        ckpt = torch.load(path)
        self.logger.info(f"Loading selection model weights from: {path}")
        
        # load selection encoder weights
        is_enc_loaded: bool = False
        for key in encoder_keys:
            try:
                self.selection_encoder.load_state_dict(ckpt[key])
                self.logger.info(f"Loaded selection encoder weights using key = `{key}`")
                is_enc_loaded = True
                break
            except KeyError as _:
                self.logger.info(f"Invalid key: `{key}`. Trying next key.")
                continue
        if not is_enc_loaded:
            self.logger.info(f"Failed to load selection encoder weights using keys from {encoder_keys}")
        
        # load selection head weights
        is_head_loaded: bool = False
        for key in head_keys:
            try:
                self.selection_head.load_state_dict(ckpt[key])
                self.logger.info(f"Loaded selection head weights using key = `{key}`")
                is_head_loaded = True
                break
            except KeyError as _:
                self.logger.info(f"Invalid key: `{key}`. Trying next key.")
                continue
        if not is_head_loaded:
            self.logger.info(f"Failed to load selection head weights using keys from {head_keys}")

    def load_outcome_weights_from_checkpoint(self,
                                             path: str,
                                             encoder_keys: typing.Iterable[str] = ['encoder', 'outcome_encoder'],
                                             head_keys: typing.Iterable[str] = ['outcome_head', ], ):
        
        ckpt = torch.load(path)
        self.logger.info(f"Loading outcome model weights from: {path}")
        
        # load selection encoder weights
        is_enc_loaded: bool = False
        for key in encoder_keys:
            try:
                self.outcome_encoder.load_state_dict(ckpt[key])
                self.logger.info(f"Loaded outcome encoder weights using key = `{key}`")
                is_enc_loaded = True
                break
            except KeyError as _:
                self.logger.info(f"Invalid key: `{key}`. Trying next key.")
                continue
        if not is_enc_loaded:
            self.logger.info(f"Failed to load outcome encoder weights using keys from {encoder_keys}")
        
        # load selection head weights
        is_head_loaded: bool = False
        for key in head_keys:
            try:
                self.outcome_head.load_state_dict(ckpt[key])
                self.logger.info(f"Loaded outcome head weights using key = `{key}`")
                is_head_loaded = True
                break
            except KeyError as _:
                self.logger.info(f"Invalid key: `{key}`. Trying next key.")
                continue
        if not is_head_loaded:
            self.logger.info(f"Failed to load outcome head weights using keys from {head_keys}")

    def _set_learning_phase(self, train: bool = True) -> None:

        if train:
            # selection encoder
            if self.args.freeze_selection_encoder:
                self.selection_encoder.eval()
            else:
                self.selection_encoder.train()
            # selection head
            if self.args.freeze_selection_head:
                self.selection_head.eval()
            else:
                self.selection_head.train()
            # outcome encoder
            if self.args.freeze_outcome_encoder:
                self.outcome_encoder.eval()
            else:
                self.outcome_encoder.train()
            # outcome head
            self.outcome_head.train()
        else:
            self.outcome_encoder.eval(); self.outcome_head.eval()
            self.selection_encoder.eval(); self.selection_head.eval()

    def _log_correlation_info(self, epoch: int, total_epochs: int) -> None:

        if self.logger is not None:
            _msg_fmt: str = f">{len(str(total_epochs))}"
            msg: str = f"(Train) Epoch [{epoch:{_msg_fmt}}/{total_epochs}]: "
            msg += "Correlation = " + ", ".join([
                f"{torch.tanh(self._rho[k]):.3f} ({actual})" for k, actual in enumerate(self.train_domains)
            ])
            self.logger.info(msg);
            if self.defaults.loss_type == 'regression':
                self.logger.info(f"Sigma: {self.sigma.item():.3f}")

    @staticmethod
    def arctanh(x: torch.Tensor, clip: typing.Optional[tuple] = (-0.9, 0.9)):
        x = torch.clip(x, min=clip[0], max=clip[1])
        return torch.atanh(x)

    def _calibrate(self, data_loader: torch.utils.data.DataLoader) -> torch.FloatTensor:
        """Temperature scaling of selection outputs."""

        # eval mode
        self.selection_encoder.eval()
        self.selection_head.eval();

        s_pred_in_probits, domain = list(), list()

        # accumulate predictions and targets
        _desc: str = f":thumbs_up: calibrating selection model... "
        for _, batch in track(enumerate(data_loader),
                              total=len(data_loader),
                              description=_desc, transient=False):

            x = batch['x'].to(self.device)
            if self.eval_transform is not None:
                x = self.eval_transform(x)

            with torch.no_grad():
                s_pred_in_probits += [self.selection_head(self.selection_encoder(x))]
                domain += [batch['domain'].to(self.device)]

        s_pred_in_probits = torch.cat(s_pred_in_probits, dim=0)  # (B, K); float
        domain = torch.cat(domain, dim=0)                        # (B,  ); long
        s_true = domain.view(-1, 1).eq(
            torch.tensor(self.train_domains, dtype=torch.long, device=self.device).view(1, -1)
        ).long()  # (B, K); long

        # find optimal temperature w.r.t data
        _nD: int = len(self.train_domains)
        temperature = nn.Parameter(torch.ones(1, _nD, dtype=torch.float32, device=self.device, requires_grad=True))
        assert temperature.requires_grad;
        optimizer = torch.optim.LBFGS([temperature], lr=.001, max_iter=10000, line_search_fn='strong_wolfe')

        def _eval_closure():
            # closure for `optim.LBFGS.step(closure)`. must return loss tensor.
            optimizer.zero_grad();
            loss = F.binary_cross_entropy(
                self._normal.cdf(s_pred_in_probits.div(temperature)),
                s_true.float(),
                reduction='mean',
            )
            loss.backward();
            return loss

        # update
        loss = optimizer.step(_eval_closure)
        if self.logger is not None:
            temp_string = ", ".join([f"{t:.4f}({d})" for t, d in zip(temperature.flatten(), self.train_domains)])
            self.logger.info(f"(Calibration) Loss: {loss.item():.4f} | Optimal temperature = {temp_string}")
            del temp_string;

        s_true_flat = s_true.flatten()

        # measure {ece, mce} before calibration
        s_prob_uncalib = self._normal.cdf(s_pred_in_probits).flatten()
        ece_before = expected_calibration_error(s_prob_uncalib, s_true_flat)
        mce_before = maximum_calibration_error(s_prob_uncalib, s_true_flat)

        # measure {ece, mce} after calibration
        s_prob_calib = self._normal.cdf(s_pred_in_probits.div(temperature)).flatten()
        ece_after = expected_calibration_error(s_prob_calib, s_true_flat)
        mce_after = maximum_calibration_error(s_prob_calib, s_true_flat)
        if self.logger is not None:
            self.logger.info(f"(Calibration) ECE = {ece_before * 100:.3f} -> {ece_after * 100:.3f}")
            self.logger.info(f"(Calibration) MCE = {mce_before * 100:.3f} -> {mce_after * 100:.3f}")

        # save reliability graph (uncalibrated)
        draw_reliability_graph(
            preds=s_prob_uncalib,
            targets=s_true_flat,
            filename=os.path.join(self.checkpoint_dir, 'before_calibration.png'),
            title='Uncalibrated',
        )

        # save reliability graph (calibrated)
        draw_reliability_graph(
            preds=s_prob_calib,
            targets=s_true_flat,
            filename=os.path.join(self.checkpoint_dir, 'after_calibration.png'),
            title=f'Calibrated with optimal temperature'
        )

        return temperature


class ContinuousHeckmanDG1(HeckmanDG):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig) -> None:
        super(ContinuousHeckmanDG1, self).__init__(args=args, defaults=defaults)


class BinaryHeckmanDG1(HeckmanDG):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig) -> None:
        super(BinaryHeckmanDG1, self).__init__(args=args, defaults=defaults)


class MultinomialHeckmanDG1(HeckmanDG):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig) -> None:
        super(MultinomialHeckmanDG1, self).__init__(args=args, defaults=defaults)


class SharedHeckmanDG1(HeckmanDG):
    def __init__(self, args: argparse.Namespace, defaults: DefaultConfig) -> None:
        super(SharedHeckmanDG1, self).__init__(args=args, defaults=defaults)

    def _init_modules(self):

        # 1. shared encoder
        self.encoder = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )

        # 2. selection head
        self.selection_head = nn.Linear(
            in_features=self.encoder.out_features,
            out_features=len(self.defaults.train_domains),
            bias=True,
        )
        self.selection_head.bias.data.fill_(0.)

        # 3. outcome head
        self.outcome_head = nn.Linear(
            in_features=self.encoder.out_features,
            out_features=NetworkInitializer._out_features[self.args.data][0],
            bias=True,
        )
        self.outcome_head.bias.data.fill_(0.)

        # 4. correlation (and sigma for regression tasks)
        self.rho: torch.FloatTensor = torch.rand(
            len(self.defaults.train_domains), dtype=torch.float32,
            requires_grad=True, device=self.device,
        )
        if self.defaults.loss_type == 'regression':
            self.sigma = torch.ones(
                1, dtype=torch.float32, requires_grad=True, device=self.device,
            )

    def _init_transforms(self) -> None:
        return super()._init_transforms()

    def _init_cuda(self) -> None:

        self.encoder.to(self.device)
        self.selection_head.to(self.device)
        self.outcome_head.to(self.device)

        if self.logger is not None:
            self.logger.info(f"Moved modules to device: {self.device}")

    def _init_optimization(self) -> None:
        """..."""

        params: typing.List[dict] = [
            {'params': self.encoder.parameters(),},
            {'params': self.selection_head.parameters(),},
            {'params': self.outcome_head.parameters(),},
        ]

        self.optimizer = configure_optimizer(
            params=params,
            name=self.args.optimizer,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        misc_params: typing.List[dict] = [{'params': self.rho},]
        if (self.defaults.loss_type == 'regression'):
            misc_params.append({'params': self.sigma})
        self.misc_optimizer = configure_optimizer(
            params=misc_params,
            name='adam',
            lr=0.01,
            weight_decay=0.001,
        )

        if self.args.scheduler is not None:
            self.scheduler = configure_learning_rate_scheduler(
                optimizer=self.optimizer,
                name=self.args.scheduler,
                epochs=self.args.epochs,
                warmup_epochs=self.args.scheduler_lr_warmup,
                min_lr=self.args.scheduler_min_lr,
            )

        if self.logger is not None:
            self.logger.info(f"Initialized optimizers and schedulers.")

    def _init_checkpoint_dir(self) -> None:
        """..."""
        self.checkpoint_dir: str = \
            f"./checkpoints/heckmanDG-1/shared/{self.args.data}/{self.args.backbone}/{self.args.hash}"
        os.makedirs(self.checkpoint_dir, exist_ok=False)

    def save_checkpoint(self, path: str, epoch: int, **kwargs) -> None:
        """Save model to a `***.pth.tar` file."""
        ckpt = {
            'encoder': self.encoder.state_dict(),
            'selection_head': self.selection_head.state_dict(),
            'outcome_head': self.outcome_head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rho': self.rho,
            'misc_optimizer': self.misc_optimizer.state_dict(),
        }

        if (self.defaults.loss_type == 'regression'):
            ckpt['sigma'] = self.sigma

        ckpt['epoch'] = epoch
        if kwargs:
            ckpt.update(kwargs)

        torch.save(ckpt, path)

    @property
    def selection_encoder(self):
        return self.encoder

    @property
    def outcome_encoder(self):
        return self.encoder
