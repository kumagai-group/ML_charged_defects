# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from cgcnn.pooling import Pooling, AveragePooling, SitePooling


@dataclass
class CgcnnFeatures:
    site_features: Tensor  # N x A
    bond_features: Tensor  # N x M x B
    bond_indices: Tensor   # N x M
    """
    N: number of atoms in all the batched structures
    A: number of embedded or convoluted site features 
    B: number of bond features
    M: number of neighbors
    """

    @property
    def N(self):
        return len(self.site_features)

    @property
    def A(self):
        return len(self.site_features[0])

    @property
    def B(self):
        return len(self.bond_features[0, 0])

    @property
    def M(self):
        return len(self.bond_features[0])

    @property
    def v_i(self):  # N x M x A
        # self site features. Copy tensor along 1st dim.
        return self.site_features.unsqueeze(1).expand(self.N, self.M, self.A)

    @property
    def v_j(self):  # N x M x A
        # self site features
        return self.site_features[self.bond_indices, :]

    @property
    def u_ij(self):  # N x M x B
        return self.bond_features

    def z_ij(self, v_j):  # N x M x (2A + B)
        return torch.cat([self.v_i, v_j, self.u_ij], dim=2)


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, A, B):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        A(int): Number of embedded site features.
        B(int): Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.A = A
        self.B = B
        self.fc = nn.Linear(2*A+B, 2*A)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*A)
        self.bn2 = nn.BatchNorm1d(A)

    def forward(self,
                site_features: Tensor,  # N x A
                bond_features: Tensor,  # N x M x B
                bond_indices: Tensor):
        """
        Forward pass

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        M = len(bond_features[0])
        N = len(site_features)
        A = len(site_features[0])

        v_i = site_features.unsqueeze(1).expand(N, M, A)
        v_j = site_features[bond_indices, :]
        u_ij = bond_features
        z_ij = torch.cat([v_i, v_j, u_ij], dim=2)

        total_gated_features = self.fc(z_ij)
        total_gated_features = self._batch_norm(M, N, total_gated_features)
        nbr_filter, nbr_core = total_gated_features.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        nbr_summed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_summed = self.bn2(nbr_summed)
        updated_site_features = self.softplus(site_features + nbr_summed)
        return updated_site_features

    def _batch_norm(self, M, N, total_gated_fea):
        # Apply batch normalization, and return it with the same size.
        return self.bn1(total_gated_fea.view(-1, self.A*2)).view(N, M, self.A*2)


class CGCNN(nn.Module):
    def __init__(self,
                 num_orig_site_fea,
                 A,
                 B,
                 n_cnn_layer,
                 hidden_dim,
                 p=0,
                 add_feature_after_pooing=True):
        """
        A: number of embedded or convoluted site features
        B: number of bond features
        """
        super(CGCNN, self).__init__()
        self.A = A
        self.fc1 = nn.Linear(num_orig_site_fea, A)
        self.cls = nn.ModuleList([ConvLayer(A, B) for _ in range(n_cnn_layer)])
        self.softplus = nn.Softplus()
        self.bn = nn.BatchNorm1d(A + add_feature_after_pooing)
        self.fc2 = nn.Linear(A + add_feature_after_pooing, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.loss_func = nn.MSELoss()

    def forward(self,
                site_features: Tensor,
                bond_features: Tensor,
                bond_indices: Tensor,
                pooling: Pooling,
                feature_after_pooing: Tensor = None):
        """
        Forward pass
        -------
        N_atom:
        N_ini:
        N_emb
        N_hidden

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        TODO: Redundant to calculate up to pooling for the same vacancy but with
        different q.

        """
        # (N_atom, N_ini) -> (N_atom, N_emb)
        embedded_site_features = self.fc1(site_features)
        for cl in self.cls:
            embedded_site_features = cl(site_features=embedded_site_features,
                                        bond_features=bond_features,
                                        bond_indices=bond_indices)

        # pooling
        features = pooling(embedded_site_features)
        if feature_after_pooing is not None:
            features = torch.cat([features, feature_after_pooing.unsqueeze(1)],
                                 dim=1)

        if len(features) > 1:
            features = self.bn(features)
        # (N_prop, N_emb) -> (N_prop, N_hidden)
        features = self.dropout(features)
        hidden_feas = self.fc2(self.softplus(features))
        #                 -> (N_prop, 1)
        hidden_feas = self.dropout(hidden_feas)
        predicted = self.fc3(self.softplus(hidden_feas))
        #                 -> N_prop
        return torch.flatten(predicted)


