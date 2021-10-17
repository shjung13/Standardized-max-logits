"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes'],
                    help='validation dataset list')
parser.add_argument('--val_interval', type=int, default=100000, help='validation interval')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--freeze_trunk', action='store_true', default=False)

parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')

parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument('--score_mode', type=str, default='msp',
                    help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=False,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=0,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=0,
                    help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=False,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=0,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=0,
                    help='kernel dilation rate of dilated smoothing')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
print("RANDOM_SEED", random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2


if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

def main():

    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_loader, val_loaders, train_obj, extra_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
        train_loader.sampler.set_epoch(epoch + 1)
        if i % args.val_interval == 0:
            for dataset, val_loader in val_loaders.items():
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
        else:
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                # if args.apex:
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        epoch += 1

    # Validation after epochs
    for dataset, val_loader in val_loaders.items():
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)


def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break
        inputs, seg_gts, ood_gts, _, aux_gts = data

        B, C, H, W = inputs.shape
        num_domains = 1
        inputs = [inputs]
        seg_gts = [seg_gts]
        ood_gts = [ood_gts]
        aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, seg_gts, ood_gts, aux_gts)):
            input, seg_gt, ood_gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None

            input, seg_gt, ood_gt = input.cuda(), seg_gt.cuda(), ood_gt.cuda()

            optim.zero_grad()

            outputs_index = 0
            outputs = net(input, seg_gts=seg_gt, ood_gts=ood_gt, aux_gts=aux_gt)
            main_loss, aux_loss, anomaly_score = outputs
            total_loss = main_loss + (0.4 * aux_loss)

            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss

            if args.local_rank == 0:
                if i % 50 == 49:
                    msg = '[epoch {}], [iter {} / {} : {}], [total loss {:0.6f}], [seg loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                        main_loss.item(), optim.param_groups[-1]['lr'],
                        time_meter.avg / args.train_batch_size)

                    logging.info(msg)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    writer.add_scalar('loss/main_loss', (main_loss.item()),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, seg_gts, ood_gts, img_names, _ = data

        assert len(inputs.size()) == 4 and len(seg_gts.size()) == 3
        assert inputs.size()[2:] == seg_gts.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs = inputs.cuda()
        seg_gts_cuda = seg_gts.cuda()

        with torch.no_grad():
            main_out, anomaly_score = net(inputs)

        del inputs

        assert main_out.size()[2:] == seg_gts.size()[1:]
        assert main_out.size()[1] == datasets.num_classes

        main_loss = criterion(main_out, seg_gts_cuda)


        val_loss.update(main_loss.item(), batch_pixel_size)

        del seg_gts_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = main_out.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([seg_gts, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), seg_gts.numpy().flatten(),
                             datasets.num_classes)
        del main_out, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()


    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    return val_loss.avg

if __name__ == '__main__':
    main()

