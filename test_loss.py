import torch
from losses import MetricLoss
import torch.nn.functional as F

cri = MetricLoss()


x = torch.ones(4, 5)

# x = F.one_hot(torch.arange(0, 5)).float()
print(x)
loss = cri.orth_reg(x)
print(loss)
