"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MetricLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, reg=False, dropout=0.1):
        super(MetricLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reg = reg
        self.drop_p = dropout
        self.cri = nn.BCELoss()

    def forward(self, x_emb, y_emb, y_pred, lab_emb, tgt):

        """
        x_emb:      [batch_size, proj_size]
        y_emb:      [batch_size, proj_size]
        y_pred:     [num_classes, num_classes]
        lab_emb:    [num_classes, proj_size]
        tgt:        [batch_size]
        """
        
        reg_loss = self.orth_reg(lab_emb)
        contrastive_loss = self.cont_loss(x_emb, y_emb.detach())
        ce_loss = self.ce_loss(y_pred)
        data = {"reg_loss": reg_loss, "cont_loss":contrastive_loss, "ce_loss":ce_loss}
        return data

    def orth_reg(self, w):
    
        w = torch.transpose(w, 1,0)
        bsz = w.size(0)
        label = torch.arange(0, w.size(0)).to(w.device)
        mask = torch.eye(bsz).to(w.device)
        prod = torch.matmul(w, w.T)
        reg_loss = F.cross_entropy(prod, label)
        return reg_loss
    
    def cont_loss(self, x, y):
        
        x = F.dropout(F.normalize(x, dim=-1), self.drop_p)
        y = F.dropout(F.normalize(y, dim=-1), self.drop_p)
        tgt = torch.ones(x.size(0), device=x.device)
        return F.cosine_embedding_loss(x, y.detach(), tgt)

    def ce_loss(self, pred):
        tgt = torch.arange(0, pred.size(-1), device=pred.device)
        return F.cross_entropy(pred, tgt)
