import torch
import torch.nn as nn
import torch.nn.functional as F


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius*2+1)**2*args.cost_heads_num*args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 128, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.convf1_ = nn.Conv2d(4, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-4, k_conv=args.k_conv)

    def forward(self, flow, corr):
        corr1, corr2 = torch.split(corr, [self.cor_planes, self.cor_planes], dim=1)
        cor = F.gelu(torch.cat([self.convc1(corr1), self.convc1(corr2)], dim=1))

        cor = self.convc2(cor)

        flo = self.convf1_(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKMotionEncoder6_Deep_nopool_res_Mem(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius * 2 + 1) ** 2 * args.cost_heads_num * args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.convf1_ = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=args.k_conv)

    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1_(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKMotionEncoder6_Deep_nopool_res_Mem_skflow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (args.corr_radius * 2 + 1) ** 2 * args.cost_heads_num * args.corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 256, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.convf1 = nn.Conv2d(2, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-2, k_conv=args.k_conv)

    def forward(self, flow, corr):
        cor = F.gelu(self.convc1(corr))

        cor = self.convc2(cor)

        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)




class SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args

        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]

        self.encoder = SKMotionEncoder6_Deep_nopool_res_Mem(args)
        self.gru = PCBlock4_Deep_nopool_res(128 + hidden_dim + hidden_dim + 128, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

        self.to_v = nn.Conv2d(128, 128, 1, bias=False)
        self.quality_head = nn.Conv2d(128, 1, 1)


    def get_motion_and_value(self, flow, corr):
        motion_features = self.encoder(flow, corr)
        return motion_features

    def forward(self, net, inp, motion_features, motion_features_global):
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))
        delta_flow = self.flow_head(net)
        flow_q = torch.sigmoid(self.quality_head(net))
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow,flow_q


class SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args

        args.k_conv = [1, 15]
        
        args.PCUpdater_conv = [1, 7]

        self.encoder = SKMotionEncoder6_Deep_nopool_res_Mem_skflow(args)
        self.gru = PCBlock4_Deep_nopool_res(128 + hidden_dim + hidden_dim + 128, 128, k_conv=args.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 2, k_conv=args.k_conv)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))
        
       

    def get_motion_and_value(self, flow, corr):
        motion_features = self.encoder(flow, corr)
        return motion_features

    def forward(self, net, inp, motion_features,motion_fuse):
        inp_cat = torch.cat([inp, motion_features,motion_fuse], dim=1)

        net = self.gru(torch.cat([net, inp_cat], dim=1))
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

   