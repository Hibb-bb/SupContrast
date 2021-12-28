"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def do_shit(self, x):
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x, layer=5):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out5 = self.avgpool(out5)
        out5 = torch.flatten(out5, 1)
        return [self.do_shit(out1), self.do_shit(out2), self.do_shit(out3), self.do_shit(out4), out5]


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.sub_heads = self._make_heads()

    def _make_heads(self):
        lst=[]
        for i in range(4):
            x = nn.Sequential(
                    nn.LazyLinear(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 128)
                    )
            lst.append(x)
        return nn.ModuleList(lst)

    def forward(self, x):
        feats = self.encoder(x)
        sub_feat = feats[:-1]
        sub_feat = [F.normalize(self.sub_heads[i](sub_feat[i]), dim=1) for i in range(len(sub_feat))]
        feat = F.normalize(self.head(feats[-1]), dim=1)
        sub_feat.append(feat)
        return sub_feat

    def encode(self, x):
        feats = self.encoder(x)
        return feats[-1]

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        self.num_classes=num_classes
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fcs = nn.ModuleList([nn.LazyLinear(num_classes)])
        nn.Linear(dim_in, num_classes)

    def forward(self, x, layer_idx=1):
        enc = self.encoder(x)
        return self.fc(enc[layer_idx])
        # return self.fc(self.encoder(x))

class ALResNet(nn.Module):
    def __init__(self, name='resnet50', num_classes=10, feat_dim=128):
        super(ALResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        # self.fc = nn.Linear(dim_in, num_classes)
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        # self.y_enc = nn.Embedding(num_classes, feat_dim)
        self.y_enc = nn.Linear(num_classes, feat_dim, bias=False)
        self.y_dec = nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes
        self.one_hot = F.one_hot(torch.arange(0, num_classes), num_classes=num_classes).float().cuda()
        self.sub_heads = self._make_heads()

    def _make_heads(self):
        lst=[]
        for _ in range(4):
            a = nn.Sequential(
                    nn.LazyLinear(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 128)
                    )
            lst.append(a)
        return nn.ModuleList(lst)

    def forward(self, x=None, y=None, mode='ml'):
        assert mode in ['ml', 'warm', 'inf']
        if mode == 'warm':
            y = F.one_hot(y).float().cuda()
            y_emb = self.y_enc(y)
            # print(y_emb.shape)
            pred = self.y_dec(y_emb)
            return pred, self.y_enc.weight
            # return pred, self.y_enc(self.one_hot.cuda())

        elif mode == 'inf':
            return self.y_dec(self.head(self.encoder(x)[-1]))
        
        elif mode == 'ml':
            """metric learning"""
            
            y = F.one_hot(y).float().cuda()
            lab_emb = self.y_enc(self.one_hot)
            x_embs = self.encoder(x)
            x_embs[-1] = self.head(x_embs[-1])
            for i in range(4):
                x_embs[i] = self.sub_heads[i](x_embs[i])
            y_emb = self.y_enc(y).detach()
            y_pred = self.y_dec(lab_emb)
            
            return x_embs, y_emb, y_pred, self.y_enc.weight

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
