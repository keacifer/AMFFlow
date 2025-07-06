import torch
import torch.nn as nn
import torch.nn.functional as F

from .AMF import AMF
from ..encoders import twins_svt_large
from .cnn import BasicEncoder
from .corr import CorrBlock
from .utils import coords_grid
from .sk2 import SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow



class AMFFlowNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = 128
        self.context_dim = 128
        

       
        self.AMF = AMF(feature_dim = 256, input_dim=128)

        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
            self.proj = nn.Conv2d(256, 256, 1)
        elif cfg.cnet == 'basicencoder':
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn='batch',init=True)
        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain) 
            self.channel_convertor = nn.Conv2d(256, 256, 1, padding=0, bias=False)
        elif cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance',init=True)
        
        if self.cfg.gma == 'GMA-SK2':
            print("[Using GMA-SK2]")
            self.cfg.cost_heads_num = 1
            self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow(args=self.cfg, hidden_dim=128)

        print("[Using corr_fn {}]".format(self.cfg.corr_fn))
        
       

    def encode_features(self, frame, flow_init=None):
        if len(frame.shape) == 5:
            need_reshape = True
            b, t = frame.shape[:2]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            need_reshape = False
        else:
            raise NotImplementedError
        fmaps = self.fnet(frame)#.float()
        fmaps = self.channel_convertor(fmaps)
        if need_reshape:
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])    
        return fmaps
            



    def encode_context(self, frame):
        if len(frame.shape) == 5:
            need_reshape = True
            b, t = frame.shape[:2]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            need_reshape = False
        else:
            raise NotImplementedError
        cnet = self.cnet(frame)
        if self.cfg.cnet == 'twins':
           cnet = self.proj(cnet)
        cnet = cnet.view(b, t, *cnet.shape[-3:])
        return cnet

    def predict_amfflow(self,cfg,images):   
        
        cmaps = self.encode_context(images)
        fmaps = self.encode_features(images)    
        B,S,C,H,W = fmaps.shape
        flow_predictions = []
        device = fmaps.device
        coords0 = self.initialize_flow_feature(fmaps[:,0]) 
        
        for ti in range(S-1):    
            flow_predictions_perframe = []
            coords1 = self.initialize_flow_feature(fmaps[:,ti])
            corr_fn = CorrBlock(fmaps[:,ti], fmaps[:,ti+1],
                            num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
           
            net, inp = torch.split(cmaps[:,ti], [self.hidden_dim, self.context_dim], dim=1)
            if ti == 0:
                self.AMF.features_mem = cmaps[:,ti]
            static_coefficient = self.AMF.memory_extraction( cmaps[:,ti])    
            
            for i in range(cfg.decoder_depth):
                coords1 = coords1.detach()
                corr = corr_fn(coords1)
                flow = coords1 - coords0
                motion_features  = self.update_block.get_motion_and_value(flow, corr)
                if ti == 0:
                    self.AMF.motion_mem   = motion_features
                motion_fuse = self.AMF.memory_fusion( motion_features,static_coefficient)
                net, up_mask, delta_flow  = self.update_block(net, inp,  motion_features, motion_fuse)
                coords1 = coords1 + delta_flow
                flow_low = coords1 - coords0
                flow_up = self.upsample_flow(flow_low, up_mask)    
                flow_predictions_perframe.append(flow_up )

            self.AMF.memory_update(cmaps[:,ti], cmaps[:,ti+1],motion_fuse)   
            flow_predictions.append(torch.stack(flow_predictions_perframe, dim=1))
        out =  torch.stack(flow_predictions, dim=2) 
        return out
        

    def initialize_flow_feature(self, feature):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        if len(feature.shape) == 5:
            B,S, C, H, W = feature.shape
        if len(feature.shape) == 4:
            B,C, H, W = feature.shape
        coords0 = coords_grid(B, H , W ).to(feature.device)
        return coords0

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)


    def forward(self,images):
        #with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        flow_predictions = self.predict_amfflow(self.cfg,images)
        return flow_predictions
