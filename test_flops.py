import torch
from thop import profile, clever_format
from torch._C import device

from networks import SupCEResNet, SupConResNet, ALResNet, resnet50

# resnet option
NAME = 'resnet50'
# (batch, channel, height, width)
N, C, H, W = 1, 3, 224, 224
# label
CLASSES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_resnet():

    model = resnet50().to(device)
    input = torch.rand(N, C, H, W).to(device)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], '%.4f')

    print(f'[INFO] resnet50 totol params: {params}\n[INFO] resnet50 totol macs: {macs}')

def test_supce():

    model = SupCEResNet(name='resnet50').to(device)
    input = torch.rand(N, C, H, W).to(device)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], '%.4f')

    print(f'[INFO] SupCE totol params: {params}\n[INFO] SupCE totol macs: {macs}')


def test_supcon():

    model = SupConResNet(name='resnet50').to(device)
    input = torch.rand(N, C, H, W).to(device)
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], '%.4f')

    print(f'[INFO] SupCon totol params: {params}\n[INFO] SupCon totol macs: {macs}')


def test_al():

    model = ALResNet(name='resnet50').to(device)
    input = torch.rand(N, C, H, W).to(device)
    output = torch.arange(CLASSES).to(device)
    macs, params = profile(model, inputs=(input, output))
    macs, params = clever_format([macs, params], '%.4f')

    print(f'[INFO] AL totol params: {params}\n[INFO] AL totol macs: {macs}')


if __name__ == '__main__':
    test_al()