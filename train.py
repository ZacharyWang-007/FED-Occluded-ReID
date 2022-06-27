import numpy as np
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
from collections import OrderedDict

from model.hm import HybridMemory
from loss.infonce import InfoNCE
import torch
from torch.cuda import amp
import torch.nn as nn
import os
import argparse
import torch.nn.functional as F
# from timm.scheduler import create_scheduler
from config import cfg
import collections

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def extract_features(model, data_loader, print_freq=50):
    model.eval()

    features = OrderedDict()
    labels = OrderedDict()

    with torch.no_grad():

        for i, (imgs, pids, _, _, _, fnames) in enumerate(data_loader):
            # with amp.autocast(enabled=True):
            outputs = model(imgs.cuda(), memory_only=True)
        
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid


            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'.format(i + 1, len(data_loader)))

    return features, labels


if __name__ == '__main__':

    print('this this this this')

    print('torch.cuda.device_count()')
    print(torch.cuda.device_count())

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/OCC_Duke/vit_base.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, dataset = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num).cuda()
    
    
    loss_func = make_loss(cfg, num_classes=num_classes)
    infonce_loss = InfoNCE(num_samples=num_classes)

    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    # Initialize source domain instance features
    print('Initialize feature centers')
    in_planes = 768
    memory = HybridMemory(in_planes * 4, num_classes, temp=0.05, momentum=0.2).cuda()
    
    features, _ = extract_features(model, train_loader_normal, )
    features_dict = collections.defaultdict(list)

    for f, pid, _, _ in sorted(dataset.train):
        features_dict[pid].append(features[f.split('/')[-1]].unsqueeze(0))
    
    feature_centers = [torch.cat(features_dict[pid], 0).mean(0) for pid in sorted(features_dict.keys())]
    feature_centers = torch.stack(feature_centers, 0)

    memory.features = feature_centers
    memory.labels = torch.arange(num_classes).cuda()

    memory2 = HybridMemory(in_planes * 4, num_classes, temp=0.05, momentum=0.2).cuda()
    memory2.features = feature_centers
    memory2.labels = torch.arange(num_classes).cuda()

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        infonce_loss,
        num_query, args.local_rank,
        memory, 
        memory2
    )
