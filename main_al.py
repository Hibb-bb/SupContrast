from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import create_toy, accuracy
from networks.resnet_big import ALResNet
from losses import MetricLoss
import numpy as np

import wandb
import os

os.environ["WANDB_SILENT"] = "true"

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--warm_ae', action='store_true', default=False)
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--freeze', action='store_true', default=False)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='AL',
            choices=['SupCon', 'SimCLR', 'AL'],  help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--loss_mode', type=str, default='unif',
            choices=['orth', 'l2_reg_ortho', 'unif', 'iso', 'vicreg'])

    opt = parser.parse_args()
    
    print('opts:')
    print('reg', opt.reg)
    print('freeze', opt.freeze)
    print('warm ae', opt.warm_ae)

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/AL/{}_models'.format(opt.dataset)
    opt.tb_path = './save/AL/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_freeze_{}_warmae_{}_reg_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.freeze, opt.warm_ae, opt.reg)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
   
    config = {
            "model":opt.model,
            "dataset":opt.dataset,
            "lr":opt.learning_rate,
            "batch_size":opt.batch_size,
            "freeze":opt.freeze,
            "warm_ae":opt.warm_ae,
            "reg":opt.reg,
            "loss mode":opt.loss_mode
            }
    wandb.init(project='SupCon_AL_label_neg',config=config, entity='al-train')

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=opt.data_folder, train=False,
                                         transform=test_transform,
                                         download=True)

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        test_dataset = datasets.CIFAR100(root=opt.data_folder, train=False,
                                          transform=test_transform,
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )

    return train_loader, test_loader


def set_model(opt):
    model = ALResNet(name=opt.model)
    criterion = MetricLoss(temperature=opt.temp, mode=opt.loss_mode)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    loss_data = {"reg_loss":[], "cont_loss":[], "ce_loss":[]}
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        # metric learning right here, bro
        # compute loss
        x_emb, y_emb, y_pred, lab_emb = model(images, labels, mode='ml')

        loss_dict = criterion(x_emb, y_emb, y_pred, lab_emb, labels)
        loss = (1-opt.reg)*loss_dict['cont_loss']
        loss += opt.reg*loss_dict['reg_loss']
        if opt.freeze is False:
            loss += loss_dict['ce_loss']
        
        # store loss data
        loss_data["reg_loss"].append(loss_dict["reg_loss"].item())
        loss_data["ce_loss"].append(loss_dict["ce_loss"].item())
        loss_data["cont_loss"].append(loss_dict["cont_loss"].item())

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg, loss_data

def test(test_loader, model):
    """test data"""
    model.eval()
    total_correct, total_sample = 0,0
    top1, top5 = [], []

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            pred = model(images, mode='inf')
        acc1, acc5 = accuracy(pred, labels, topk=(1,5))
        top1.append(acc1[0].item())
        top5.append(acc5[0].item())
        # raise Exception('acc')
        total_correct += (pred.argmax(-1) == labels).sum().item()
        total_sample += images.size(0)
    
    return total_correct/total_sample, np.mean(top1), np.mean(top5)

def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)
    # wandb.watch_model

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    print('logging at', opt.tb_folder)

    if opt.warm_ae:
        warm_loader = create_toy(100)
        
        for i in range(50):
            for data in warm_loader:
                
                optimizer.zero_grad()
                data = data[0].cuda(non_blocking=True)
                pred, lab_emb = model(y=data, mode='warm')
                ce_loss = criterion.ce_loss(pred)
                reg_loss = criterion.orth_reg(lab_emb)
                loss = ce_loss + opt.reg*reg_loss

                loss.backward()
                optimizer.step()
            wandb.log({'warm_up_ce_loss':ce_loss})
            wandb.log({'warm_up_reg_loss':reg_loss})

    if opt.freeze == True:
        for param in model.y_enc.parameters():
            param.requires_grad = False
        for param in model.y_dec.parameters():
            param.requires_grad = False

    best_acc, best_epoch = -1, 0

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, loss_data = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        test_acc, top1, top5 = test(test_loader, model)
        
        # tensorboard logger
        
        wandb.log({
            "loss":loss,
            "reg_loss":np.mean(loss_data['reg_loss']),
            "ce_loss":np.mean(loss_data['ce_loss']),
            "cont_loss":np.mean(loss_data['cont_loss']),
            "learning_rate":optimizer.param_groups[0]['lr'],
            "test_acc":test_acc*100,
            "test top1":top1,
            "test top5":top5
            })
        '''
        logger.log_value('loss', loss, epoch)
        logger.log_value('reg_loss', np.mean(loss_data['reg_loss']), epoch)
        logger.log_value('ce_loss', np.mean(loss_data['ce_loss']), epoch)
        logger.log_value('cont_loss', np.mean(loss_data['cont_loss']), epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logger.log_value('test_acc', test_acc, epoch)
        '''

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    print('best acc', best_acc)
    print('best epoch', best_epoch)

if __name__ == '__main__':
    main()
