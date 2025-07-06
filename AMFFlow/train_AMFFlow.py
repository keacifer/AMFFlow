from __future__ import print_function, division
import argparse
import numpy as np
from pathlib import Path
import torchvision.transforms as T

import torch
import torch.nn as nn
import core.datasets_video as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger
import random
from core.Networks import build_network
import os
import evaluate_AMFFlow
try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    
    loss_func = sequence_loss

    model = nn.DataParallel(build_network(cfg) )
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    
    model.cuda()
    #print(model)

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cuda')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            model.load_state_dict(ckpt_model, strict=cfg.load_strict)
        else:
            print('pretrained')
            model.module.load_state_dict(ckpt_model, strict=cfg.load_strict)


    if 'freeze_module' in cfg and cfg.freeze_module:
        print("[Freeze some modules]")
        for param in model.module.cnet.parameters():
            param.requires_grad = True
        for param in model.module.fnet.parameters():
            param.requires_grad = False
        for param in model.module.update_block.parameters():
            param.requires_grad = True
    
    model.cuda().train()
    
    

    if cfg.eval_only:
        for val_dataset in cfg.validation:
            results = {}
            if val_dataset == 'sintel_train':
                results.update(evaluate_AMFFlow.validate_sintel(model.module, cfg))
            elif val_dataset == 'spring_train':
                results.update(evaluate_AMFFlow.validate_spring(model.module, cfg))
            elif val_dataset == 'kitti_train':
                results.update(evaluate_AMFFlow.validate_kitti(model.module, cfg))
            elif val_dataset == 'sintel_test':
                results.update(evaluate_AMFFlow.create_sintel_submission(model.module, cfg))
            
            print(results)
        return

   
    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    epoch = 0
    if cfg.restore_steps > 1:
        print("[Loading optimizer from {}]".format(cfg.restore_ckpt))
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.total_steps = cfg.restore_steps - 1
        total_steps = cfg.restore_steps
        epoch = ckpt['epoch']
        for _ in range(total_steps):
            scheduler.step()

    should_keep_training = True
    while should_keep_training:

        epoch += 1
        for i_batch, data_blob in enumerate(train_loader):

            optimizer.zero_grad()
            images, flows, valids = [x.cuda() for x in data_blob]
            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                images = (images + stdv * torch.randn(*images.shape).cuda()).clamp(0.0, 255.0)
                ramdom_kernel = random.randint(2,3)*2-1
                gaussianblur = T.GaussianBlur(kernel_size = ramdom_kernel,sigma = (0.01,1.5))
                b,t,c,h,w = images.shape
                images = gaussianblur(images.view(b*t,c,h,w))
                images = images.view(b,t,c,h,w)

            output = {}
            images = 2 * (images / 255.0) - 1.0

            flow_predictions = model(images)
            flow_predictions = flow_predictions.permute(1,0,2,3,4,5)

            loss, metrics,_ = loss_func(flow_predictions, flows, valids, cfg)
            #print(loss)
            scaler.scale(loss).backward()       
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)
    
            if total_steps % cfg.val_freq == cfg.val_freq - 1 :
                print('start validation')
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1, cfg.name)
                torch.save({
                    'iteration': total_steps,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'model': model.module.state_dict(),
                }, PATH)
            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                results = {}
                
                for val_dataset in cfg.validation:
                    if val_dataset == 'sintel_train':
                        results.update(evaluate_AMFFlow.validate_sintel(model, cfg,step = total_steps))
                        logger.push(results)
                    elif val_dataset == 'kitti':
                        results.update(evaluate_AMFFlow.validate_kitti(model, cfg))
                        print("not yet im")
                    elif val_dataset == 'spring_subset_val':
                        results.update(evaluate_AMFFlow.validate_spring(model ,cfg, split='subset_val'))
                        print("not yet im")
                model.train()
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
   
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AMFFlow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--GPU_ids', type=str, default='0,1,2,3')
    parser.add_argument('--eval_only', action='store_true', default= True, help='eval only')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things_AMFFlow import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel_AMFFlow  import get_cfg
   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    args.world_size = 1

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    if not cfg.eval_only:
        loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    train(cfg)
