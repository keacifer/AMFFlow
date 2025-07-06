import torch
import torch.nn as nn
from .attention import Attention


class AMF(nn.Module):
    def __init__(self, feature_dim = 256, input_dim=128):
        super(AMF, self).__init__()
        self.features_mem = None
        self.motion_mem = None

        
        
        self.self_attn = Attention(input_dim = feature_dim,heads = 1,dim_head = feature_dim)
        self.mem_attn = Attention(input_dim = feature_dim,heads = 1,dim_head = feature_dim)
        
        self.conv_out_weight = nn.Sequential(
            nn.Conv2d(2*input_dim, 1*input_dim, 1,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1*input_dim,1*input_dim, 1,bias = False),
            nn.Sigmoid()
            )
        self.conv_zip_ch = nn.Sequential(
            nn.Conv2d(input_dim, 4*input_dim, 1,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*input_dim, input_dim, 1,bias = False),
            
            )
        self.update_attn = Attention(input_dim = feature_dim,heads = 1,dim_head = feature_dim)
        self.conv_out_mem = nn.Sequential(
            nn.Conv2d(feature_dim, 4*feature_dim, 1,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*feature_dim, feature_dim, 1,bias = False)
            )
    

    def memory_extraction(self,feature0):
        b,c,h,w = feature0.shape
        self_attn_prob  = self.self_attn(feature0,feature0)
        mem_attn_prob  = self.mem_attn(feature0,self.features_mem)
        return [self_attn_prob,mem_attn_prob]
        

    
    def memory_fusion(self,motion,static_coefficient):
        b,c,h,w = motion.shape
        motion_self = torch.matmul(static_coefficient[0],(motion).view(b, c ,h * w).permute(0, 2, 1) ).view(b, h, w, c).permute(0, 3, 1, 2)
        motion_mem =  torch.matmul(static_coefficient[1],(self.motion_mem).view(b, c ,h * w).permute(0, 2, 1)).view(b, h, w, c).permute(0, 3, 1, 2)
        #motion_all = torch.cat([motion,motion_self+motion_mem],dim =1)
        #weight = self.conv_out_weight(motion_all)
        
        motion_fuse = motion + self.conv_zip_ch(motion_self+motion_mem)
        return motion_fuse 

    def memory_update(self, feature0,feature1,motion):
        b,c,h,w = feature1.shape
        mem_attn = self.update_attn(feature0,feature1)
        message = torch.matmul(mem_attn,(feature1.view(b, c ,h * w).permute(0, 2, 1)) ).view(b, h, w, c).permute(0, 3, 1, 2)
        self.features_mem =  feature0 + self.conv_out_mem(feature0 +message)
        self.motion_mem = motion