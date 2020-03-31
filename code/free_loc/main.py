import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets.factory import get_imdb
from custom import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0


def plot_random(model, val_loader):
    denorm = transforms.Lambda(denormalize)
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        output_sig = F.sigmoid(output)
        n, m = output_sig.size(2), output_sig.size(3)

        output_imgs = output_sig.cpu().data.numpy()
        input_imgs = input.numpy()
        input_all = input_imgs
        bs, ch, h, w = input_all.shape

        for j in range(20):
            input_imgs[j] = denorm(input_all[j])
            vis.image(input_imgs[j], opts=dict(title='Image', caption='random' + format(j, '02d') + '_image'))

            gt_cls = [i for i, x in enumerate(target[j]) if x == 1]
            for k in range(len(gt_cls)):
                tmp = output_imgs[j][gt_cls[k]]
                tmp = sci.imresize(tmp, (h, w))
                vis.heatmap(tmp, opts=dict(title='random' + format(j, '02d') + '_heatmap_' + CLASS_NAMES[gt_cls[k]]))
        break


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer

    criterion = nn.BCELoss().cuda()                         #Check which loss function to apply
    # criterion = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)




    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()
    if args.vis:
        import visdom

    log_dir = '../model/'
    logger = Logger(log_dir, name = 'freeloc')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, num_cls, model, criterion, optimizer, epoch, logger)

        # evaluate on validation set
        if epoch%args.eval_freq==0 or epoch==args.epochs-1:
            m1, m2 = validate(val_loader, num_cls, model, criterion, epoch, logger)
            score = np.mm(m1,m2)
            # remember best prec@1 and save checkpoint
            
            is_best =  score > best_prec1
            best_prec1 = max(score, best_prec1)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
    
    plot_random(model, val_loader)


#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        input_var = torch.autograd.Variable(input_var, requires_grad=True)
        target_var = torch.autograd.Variable(target_var)
        bs = input.size(0)

        heatmap = model(input_var)
        n, m = heatmap.size(2), heatmap.size(3)
        output = F.max_pool2d(heatmap, kernel_size=(n,m))
        output = torch.squeeze(output)
        imoutput = F.sigmoid(output)

        heatmap = F.sigmoid(heatmap)


        # compute loss
        loss = criterion(imoutput, target_var)
        # loss = F.binary_cross_entropy_with_logits(imoutput, target_var)


        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # TODO:
        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals

        if i % args.print_freq == 0:
            global_step = epoch * steps_per_epoch + i
            logger.scalar_summary('train/loss', loss, global_step)
            logger.scalar_summary('train/metric1', avg_m1.val, global_step)
            logger.scalar_summary('train/metric2', avg_m2.val, global_step)

        if i % (steps_per_epoch // 4) == steps_per_epoch//4 - 1:

            global_step = epoch * steps_per_epoch + i
            logger.model_param_histo_summary(model, global_step)

            if epoch == 0 or epoch >= args.epochs - 1:  # just to accelerate speed
                print('draw~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                input_imgs = input.numpy()
                bs, ch, h, w = input_imgs.shape
                print(input_imgs.shape)
                output_imgs =heatmap.cpu().data.numpy()
                heatmap_all = np.ones((1, bs*h, w))
                input_all = np.ones((ch, bs*h, w))
                for j in range(bs):
                    gt_cls = [k for k, x in enumerate(target[j]) if x == 1]
                    tmp = output_imgs[j][gt_cls[0]]
                    tmp = sci.imresize(tmp, (h, w))
                    heatmap_all[:, j*h:(j+1)*h, :] = tmp
                    input_all[:, j*h:(j+1)*h, :] = input_imgs[j]

                logger.image_summary('train/images', [input_all], global_step)
                logger.image_summary('train/heatmaps', heatmap_all, global_step)


                denorm = transforms.Lambda(denormalize)
                for j in range(bs):
                    caption = format(epoch, '02d') + '_' + format(i, '03d') + '_' + format(j, '02d') + '_image'
                    tmp = denorm(input_imgs[j])
                    vis.image(tmp, opts=dict(title='Image', caption=caption))
                    gt_cls = [k for k, x in enumerate(target[j]) if x == 1]
                    for k in range(len(gt_cls)):
                        tmp = output_imgs[j][gt_cls[k]]
                        tmp = sci.imresize(tmp, (h, w))
                        caption = format(epoch, '02d') + '_' + format(i, '03d') + '_' + format(j, '02d') + '_heatmap_' + CLASS_NAMES[gt_cls[k]]
                        vis.heatmap(tmp, opts=dict(title=caption))



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``

        bs = input.size(0)
        heatmap = model(input_var)
        n, m = heatmap.size(2), heatmap.size(3)
        output = F.max_pool2d(heatmap, kernel_size=(n,m))
        output = torch.squeeze(output)
        imoutput = F.sigmoid(output)
        heatmap = F.sigmoid(heatmap)

        # compute loss
        loss = criterion(imoutput, target_var)


        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals
    logger.scalar_summary('validation/metric1', avg_m1.avg, epoch)
    logger.scalar_summary('validation/metric2', avg_m2.avg, epoch)

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='alexnet_normal.h5'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'alexnet_checkpoint.h5')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    ap_all = []
    num = num_cls
    
    for i in range(num_cls):
        target_cls = target[:, i]
        output_cls = output[:, i]
        output_cls -= 1e-5 * target_cls
        ap = sklearn.metrics.average_precision_score(target_cls, output_cls, average=None)
        if math.isnan(ap):
            ap = 0
            num -= 1
        ap_all.append(ap)
    ap_mean = np.sum(ap_all) / float(num)
    
    return ap_mean


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed

    , num_cls = target.shape
    k = 5
    target = target.cpu().numpy()
    count = 0
    tempcount = 0
    
    for i in range(bs):
    
        top_idx = np.argsort(output[i,:])[0:k]
        count = count + np.sum(target[i,:])
        count_tmp = target[i, top_idx]
        tempcount = tempcount + np.sum(count_tmp)
    
    return float(tempcount) / count


if __name__ == '__main__':
    main()
