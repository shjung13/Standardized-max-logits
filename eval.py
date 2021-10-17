"""
Evaluation Scripts
"""
from __future__ import absolute_import
from __future__ import division
from collections import namedtuple, OrderedDict
from network import mynn
import argparse
import logging
import os
import torch
import time
import numpy as np

from config import cfg, assert_and_infer_cfg
import network
import optimizer
from ood_metrics import fpr_at_95_tpr
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torchvision.transforms as standard_transforms

dirname = os.path.dirname(__file__)
pretrained_model_path = os.path.join(dirname, 'pretrained/r101_os8_base_cty.pth')

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='possible datasets for statistics; cityscapes')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')

parser.add_argument('--snapshot', type=str, default=pretrained_model_path)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

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
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

parser.add_argument('--ood_dataset_path', type=str,
                    default='/home/nas1_userB/dataset/ood_segmentation/fishyscapes',
                    help='OoD dataset path')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument('--score_mode', type=str, default='standardized_max_logit',
                    help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4,
                    help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                    help='kernel dilation rate of dilated smoothing')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

args.world_size = 1

print(f'World Size: {args.world_size}')
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

def get_net():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    net = network.get_net(args, criterion=None, criterion_aux=None)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, None, None,
                            args.snapshot, args.restore_optimizer)
        print(f"Loading completed. Epoch {epoch} and mIoU {mean_iu}")
    else:
        raise ValueError(f"snapshot argument is not set!")

    class_mean = np.load(f'stats/{args.dataset}_mean.npy', allow_pickle=True)
    class_var = np.load(f'stats/{args.dataset}_var.npy', allow_pickle=True)
    net.module.set_statistics(mean=class_mean.item(), var=class_var.item())

    torch.cuda.empty_cache()
    net.eval()

    return net

def preprocess_image(x, mean_std):
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)

    x = x.cuda()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x


if __name__ == '__main__':
    net = get_net()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ood_data_root = args.ood_dataset_path
    image_root_path = os.path.join(ood_data_root, 'leftImg8bit_trainvaltest/leftImg8bit/val')
    mask_root_path = os.path.join(ood_data_root, 'gtFine_trainvaltest/gtFine/val')

    if not os.path.exists(image_root_path):
        raise ValueError(f"Dataset directory {image_root_path} doesn't exist!")

    anomaly_score_list = []
    ood_gts_list = []

    cities = os.listdir(image_root_path)
    for city in cities:
        city_path = os.path.join(image_root_path, city)
        city_mask_path = os.path.join(mask_root_path, city)
        for image_file in tqdm(os.listdir(city_path)):
            mask_file = image_file.replace('leftImg8bit.png', 'gtFine_labelIds.png')
            image_path = os.path.join(city_path, image_file)
            mask_path = os.path.join(city_mask_path, mask_file)

            # 3 x H x W
            image = np.array(Image.open(image_path).convert('RGB')).astype('uint8')

            mask = Image.open(mask_path)
            ood_gts = np.array(mask)

            ood_gts_list.append(np.expand_dims(ood_gts, 0))

            with torch.no_grad():
                image = preprocess_image(image, mean_std)
                main_out, anomaly_score = net(image)
            del main_out

            anomaly_score_list.append(anomaly_score.cpu().numpy())

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # drop void pixels
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = -1 * anomaly_scores[ood_mask]
    ind_out = -1 * anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print('Measuring metrics...')

    fpr, tpr, _ = roc_curve(val_label, val_out)

    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(val_label, val_out)
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    print(f'AUROC score: {roc_auc}')
    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr}')

