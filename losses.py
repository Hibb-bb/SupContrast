"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, temperature=0.07, base_temperature=0.07, reg=False, dropout=0.1, mode='unif', end2end=False):
        super(MetricLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reg = reg
        self.drop_p = dropout
        self.cri = nn.BCELoss()
        self.end2end = end2end
        self.mode = 'unif'

    def forward(self, x_emb, y_emb, y_pred, lab_emb, tgt):

        """
        x_emb:      [batch_size, proj_size]
        y_emb:      [batch_size, proj_size]
        y_pred:     [num_classes, num_classes]
        lab_emb:    [num_classes, proj_size]
        tgt:        [batch_size]
        """
        if self.mode == 'orth':
            reg_loss = self.orth_reg(lab_emb)
        elif self.mode == 'unif':
            reg_loss = self.unif_reg(lab_emb)
        elif self.mode == 'l2_reg_ortho':
            reg_loss = self.l2_reg_ortho(lab_emb)
        elif self.mode == 'iso':
            reg_loss = self.isotropy(lab_emb)

        if self.end2end:
            # contrastive_loss = self.vicreg(x_emb, y_emb)
            contrastive_loss = self.cont_loss(x_emb, y_emb) + self.neg_label(x, tgt, lab_emb, lab_emb.size(1))
        else:
            # contrastive_loss = self.vicreg(x_emb, y_emb)
            contrastive_loss = self.cont_loss(x_emb, y_emb.detach()) + self.neg_label(x_emb, tgt, lab_emb, lab_emb.size(1))
        
        ce_loss = self.ce_loss(y_pred)
        data = {"reg_loss": reg_loss, "cont_loss":contrastive_loss, "ce_loss":ce_loss}
        return data

    def vicreg(self, z_a, z_b):
    
        # invariance loss
        sim_loss = F.mse_loss(z_a, z_b)
        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        # covariance loss
        N, D = z_a.size()
 
        z1 = z_a - z_a.mean(dim=0)
        z2 = z_b - z_b.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D

        return 1*cov_loss + 25*std_loss + 25*sim_loss

    def align(self, x, y, alpha=2):
        # x = F.normalize(x, dim=-1)
        # y = F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def orth_reg(self, w):
 
        w = torch.transpose(w, 1,0)
        bsz = w.size(0)
        label = torch.arange(0, w.size(0)).to(w.device)
        mask = torch.eye(bsz).to(w.device)
        prod = torch.matmul(w, w.T)
        reg_loss = F.cross_entropy(prod, label)
        return reg_loss
   
    def cont_loss2(self, x, y):
        # print(x.shape, y.shape) 
        logits = (x*y).sum(-1)
        # logits = torch.div(torch.matmul(x.T, y) , self.temperature)
        tgt = torch.ones_like(logits).to(x.device)
        loss = F.binary_cross_entropy(torch.sigmoid(logits), tgt)
        return loss

    def cont_loss(self, x, y):
        
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        tgt = torch.ones(x.size(0), device=x.device)
        return F.cosine_embedding_loss(x, y.detach(), tgt)

    def neg_label(self, x, label, lab_emb, class_num):
        # x: batch, hid_dim
        # lab_emb: class_num, hid_dim
        x = F.normalize(x, dim=-1)
        lab_emb = F.normalize(lab_emb, dim=0)
        prod = torch.matmul(x, lab_emb) # batch, class_num
        for i in range(label.size(0)):
            prod[i][label[i]] = 0
        prod = prod.mean().mean()
        return prod

    def ce_loss(self, pred):
        tgt = torch.arange(0, pred.size(-1), device=pred.device)
        return F.cross_entropy(pred, tgt)

    def unif_reg(self, x, t=2):
        x = torch.transpose(x, 1,0)
        x = F.normalize(x, dim=-1)
        return (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()+ 4)/4

    def l2_reg_ortho(self, W):

        l2_reg = None
        W = torch.transpose(W, 1,0)
        cols, rows = W.shape
        w1 = W.view(-1,cols)
        wt = torch.transpose(w1,0,1)
        m  = torch.matmul(wt,w1)
        ident = Variable(torch.eye(cols,cols))
        ident = ident.cuda()
        w_tmp = (m - ident)
        height = w_tmp.size(0)
        u = F.normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
        v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
        u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
        sigma = torch.dot(u, torch.matmul(w_tmp, v))
        l2_reg = (sigma)**2
        return l2_reg
    
    def isotropy(self, embeddings):
        """
        Computes isotropy score.
        Defined in Section 5.1, equations (7) and (8) of the paper.
        Args:
            embeddings: word vectors of shape (n_type_of_data, n_dimensions)
        Returns:
            float: isotropy score
        """
        min_z = math.inf
        max_z = -math.inf

        eigen_values, eigen_vectors = torch.linalg.eig(torch.matmul(embeddings.T, embeddings))
        
        for i in range(eigen_vectors.shape[1]):
            z_c = torch.matmul(embeddings, eigen_vectors[:, i].unsqueeze(1))
            z_c = torch.exp(z_c)
            z_c = torch.sum(z_c)
            min_z = torch.min(z_c, min_z)
            max_z = torch.max(z_c, max_z)

        return min_z/max_z

