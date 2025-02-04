import math
import torch
import numpy as np

from src.utils.utils import l2_normalize
from src.objectives.simclr import SimCLRObjective, SimCLRSupConObjective, SimCLRObjective3View
from src.objectives.infonce import NoiseConstrastiveEstimation


class AdversarialSimCLRLoss(object):

    def __init__(
        self,
        embs1,
        embs2,
        t=0.07,
        view_maker_loss_weight=1.0,
        **kwargs
    ):
        '''Adversarial version of SimCLR loss.
        
        Args:
            embs1: embeddings of the first views of the inputs
            embs1: embeddings of the second views of the inputs
            t: temperature
            view_maker_loss_weight: how much to weight the view_maker loss vs the encoder loss
        '''
        self.embs1 = embs1
        self.embs2 = embs2
        self.t = t
        self.view_maker_loss_weight = view_maker_loss_weight

        self.normalize_embeddings()

    def normalize_embeddings(self):
        self.embs1 = l2_normalize(self.embs1)
        self.embs2 = l2_normalize(self.embs2)

    def get_loss(self):
        '''Return scalar encoder and view-maker losses for the batch'''
        simclr_loss = SimCLRObjective(self.embs1, self.embs2, self.t)
        encoder_loss = simclr_loss.get_loss()
        view_maker_loss = -encoder_loss * self.view_maker_loss_weight
        return encoder_loss, view_maker_loss

class AdversarialSimCLRSupConLoss(object):

    def __init__(
        self,
        embs1,
        embs2,
        t=0.07,
        view_maker_loss_weight=1.0,
        **kwargs
    ):
        '''Adversarial version of SimCLR loss.
        
        Args:
            embs1: embeddings of the first views of the inputs
            embs1: embeddings of the second views of the inputs
            t: temperature
            view_maker_loss_weight: how much to weight the view_maker loss vs the encoder loss
        '''
        self.embs1 = embs1
        self.embs2 = embs2
        self.t = t
        self.view_maker_loss_weight = view_maker_loss_weight

        self.normalize_embeddings()

    def normalize_embeddings(self):
        self.embs1 = l2_normalize(self.embs1)
        self.embs2 = l2_normalize(self.embs2)

    def get_loss(self):
        '''Return scalar encoder and view-maker losses for the batch'''
        simclr_loss = SimCLRSupConObjective(self.embs1, self.embs2, self.t)
        encoder_loss = simclr_loss.get_loss()
        view_maker_loss = -encoder_loss * self.view_maker_loss_weight
        return encoder_loss, view_maker_loss

class AdversarialSimCLR3ViewLoss(object):

    def __init__(
        self,
        embs1,
        embs2,
        embs3,
        t=0.07,
        view_maker_loss_weight=1.0,
        alpha=0.07,
        **kwargs
    ):
        '''Adversarial version of SimCLR loss.
        
        Args:
            embs1: embeddings of the first views of the inputs
            embs2: embeddings of the second views of the inputs
            embs3: embeddings of the third views of the inputs
            t: temperature
            view_maker_loss_weight: how much to weight the view_maker loss vs the encoder loss
        '''
        self.embs1 = embs1
        self.embs2 = embs2
        self.embs3 = embs3
        self.t = t
        self.alpha = alpha
        self.view_maker_loss_weight = view_maker_loss_weight

        self.normalize_embeddings()

    def normalize_embeddings(self):
        self.embs1 = l2_normalize(self.embs1)
        self.embs2 = l2_normalize(self.embs2)
        self.embs3 = l2_normalize(self.embs3)

    def get_loss(self):
        '''Return scalar encoder and view-maker losses for the batch'''
        simclr_loss = SimCLRObjective3View(self.embs1, self.embs2, self.embs3, self.t, alpha=self.alpha)
        encoder_loss = simclr_loss.get_loss()
        view_maker_loss = -encoder_loss * self.view_maker_loss_weight
        return encoder_loss, view_maker_loss
    
    def info_nce_loss(z1, z2, temperature=0.5):
        logits = z1 @ z2.T
        logits /= temperature
        n = z2.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

class AdversarialNCELoss(object):

    def __init__(
        self,
        indices,
        outputs,
        memory_bank,
        k=4096,
        t=0.07,
        m=0.5,
        view_maker_loss_weight=1.0,
        **kwargs
    ):
        self.k, self.t, self.m = k, t, m
        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)

        self.view_maker_loss_weight = view_maker_loss_weight
        self.memory_bank = memory_bank
        self.device = outputs.device
        self.data_len = memory_bank.size

    def updated_new_data_memory(self):
        data_memory = self.memory_bank.at_idxs(self.indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * self.outputs
        return l2_normalize(new_data_memory, dim=1)

    def get_loss(self):
        nce_loss = NoiseConstrastiveEstimation(
            self.indices, self.outputs, self.memory_bank,
            k=self.k, t=self.t, m=self.m,
        )
        encoder_loss = nce_loss.get_loss()
        view_maker_loss = -encoder_loss * self.view_maker_loss_weight
        return encoder_loss, view_maker_loss
