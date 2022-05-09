import math
import torch
import numpy as np

from src.utils.utils import l2_normalize
import torch


class SimCLRObjective(torch.nn.Module):

    def __init__(self, outputs1, outputs2, t, base_t=0.07, push_only=False):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t
        self.base_t = base_t
        self.push_only = push_only

    def get_loss(self):
        # batch_size = self.outputs1.size(0)  # batch_size x out_dim
        # witness_score = torch.sum(self.outputs1 * self.outputs2, dim=1)
        # if self.push_only:
        #     # Don't pull views together.
        #     witness_score = 0
        # outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        # witness_norm = self.outputs1 @ outputs12.T
        # witness_norm = torch.logsumexp(witness_norm / self.t, dim=1) - math.log(2 * batch_size)
        # loss = -torch.mean(witness_score / self.t - witness_norm)
        # return loss
        batch_size = self.outputs1.shape[0]

        contrast_count = 2
        contrast_feature = torch.cat([self.outputs1, self.outputs2], dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = torch.eye(batch_size, dtype=torch.float32).to(contrast_feature.device)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(contrast_feature.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.t / self.base_t) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss