from __future__ import print_function, division
import argparse
from loguru import logger as loguru_logger
import random
from core.Networks import build_network
import sys
sys.path.append('core')
from PIL import Image
import os
import numpy as np
import torch
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder
from inference import inference_core_AMFFlow as inference_core
import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def inference(cfg):
    model = build_network(cfg).cuda()
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        ckpt = torch.load(cfg.restore_ckpt, map_location= device)
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=False)
        else:
            model.load_state_dict(ckpt_model, strict=False)
    model.cuda()
    model.eval()

    print(f"preparing image...")
    print(f"Input image sequence dir = {cfg.seq_dir}")
    image_list = sorted(os.listdir(cfg.seq_dir))
    imgs = [frame_utils.read_gen(os.path.join(cfg.seq_dir, path)) for path in image_list]
    imgs = [np.array(img).astype(np.uint8) for img in imgs]
    imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
    images = torch.stack(imgs)#50, 3, 436, 1024

    processor = inference_core.InferenceCore(model, config=cfg)
    images = images.cuda().unsqueeze(0)#1, 50, 3, 436, 1024
    padder = InputPadder(images.shape)
    images = padder.pad(images)
    frames = cfg.input_frames
    images = 2 * (images / 255.0) - 1.0
    
    results = []
    print(f"start inference...")
    print(f"inference frames {images.shape[1]}")
   
    for ti in range(images.shape[1] - frames+1):
        flow_pre = processor.step(images[:, ti:ti + frames])
        flow_pre = padder.unpad(flow_pre[0]).cpu()
        results.append(flow_pre)
        
    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)

    print(f"save results...")
    
    N = len(results)
    for idx in range(N):
       
        flow_img = flow_viz.flow_to_image(results[idx][-1][-1].permute(1, 2, 0).numpy())       
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(cfg.vis_dir, idx + 1, idx + 2))
    for idx in range(N):
        frame_utils.writeFlow('{}/flow_{:04}_to_{:04}.flo'.format(cfg.flo_dir, idx + 1, idx + 2))
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AMFFlow', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    parser.add_argument('--flo_dir', default='default')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things_AMFFlow import get_cfg
    

    cfg = get_cfg()
    cfg.update(vars(args))

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    inference(cfg)
#python inference.py --stage things --restore_ckpt ./ckpts/20000_TFlow.pth --seq_dir /home/zgb/cui/datasets/Sintel/training/clean/bamboo_2 --vis_dir ./viewresult --flo_dir ./flo