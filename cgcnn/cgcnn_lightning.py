# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.cgcnn_module import CGCNN
from cgcnn.collate import BatchedCgcnnInfo
from cgcnn.featurizer import BondFeaturizer, ElementFeaturizer
from cgcnn.parameters import HyperParams
from cgcnn.pooling import SitePooling


class CGCNNLightning(pl.LightningModule):
    def __init__(self,
                 params: HyperParams,
                 bond_featurizer_: BondFeaturizer,
                 elem_featurizer_: ElementFeaturizer):

        super(CGCNNLightning, self).__init__()
        self.model = CGCNN(num_orig_site_fea=elem_featurizer_.num_feature,
                           A=params.A,
                           B=bond_featurizer_.B,
                           n_cnn_layer=params.num_cnn,
                           hidden_dim=params.hidden_dim,
                           add_feature_after_pooing=True
                           )
        self.mse_loss_func = nn.MSELoss()
        self.learning_rate = params.learning_rate
        self.optimizer_class = params.optimizer
        self.milestones = params.milestones
        self.train_preds = []
        self.train_truth = []
        self.val_preds = []
        self.val_truth = []

    def forward(self, batch: BatchedCgcnnInfo):
        batched = batch.batched_cgcnn_features
        pooling = SitePooling(batch.target_site_indices_in_batch)
        return self.model(batched.site_features,
                          batched.bond_features,
                          batched.bond_indices,
                          pooling,
                          torch.tensor(batch.site_charges_in_batch))

    def step(self, batch: BatchedCgcnnInfo, type_: str):
        output = self.forward(batch)
        targets = torch.tensor(batch.target_vals_in_batch)

        loss = self.mse_loss_func(output, targets)
        self.log(f"{type_}_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(),
                                         lr=self.learning_rate)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones)
        return [optimizer], [scheduler]
