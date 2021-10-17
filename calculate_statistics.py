"""
Calculate statistics
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
                    help='a list consists of cityscapes')
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

    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):
    calculate_statistics(train_loader, net)

def calculate_statistics(train_loader, net):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    return:
    """
    net.eval()

    pred_list = None
    max_class_mean = {}
    print("Calculating statistics...")
    for i, data in enumerate(train_loader):
        inputs = data[0]

        inputs = inputs.cuda()
        B, C, H, W = inputs.shape
        batch_pixel_size = C * H * W

        with torch.no_grad():
            outputs, _ = net(inputs)

        if pred_list is None:
            pred_list = outputs.data.cpu()
        else:
            pred_list = torch.cat((pred_list, outputs.cpu()), 0)
        del outputs

        if i % 50 == 49 or i == len(train_loader) - 1:
            pred_list = pred_list.transpose(1, 3)
            pred_list, prediction = pred_list.max(3)

            class_max_logits = []
            mean_dict, var_dict = {}, {}
            for c in range(datasets.num_classes):
                max_mask = pred_list[prediction == c]
                class_max_logits.append(max_mask)

                mean = class_max_logits[c].mean(dim=0)
                var = class_max_logits[c].var(dim=0)

                mean_dict[c] = mean.item()
                var_dict[c] = var.item()

            print(f"class mean: {mean_dict}")
            print(f"class var: {var_dict}")
            np.save(f'stats/{args.dataset[0]}_mean.npy', mean_dict)
            np.save(f'stats/{args.dataset[0]}_var.npy', var_dict)

            return None

if __name__ == '__main__':
    main()

