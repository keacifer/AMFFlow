import torch

MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    B, N, _, H, W = flow_gt.shape


    NAN_flag = False

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=2).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
        flow_pre = flow_preds[i]
        
        i_loss = (flow_pre - flow_gt).abs()

        if torch.isnan(i_loss).any():
            NAN_flag = True

        _valid = valid[:, :, None]
        if cfg.filter_epe:
            loss_mag = torch.sum(i_loss**2, dim=2).sqrt()
            mask = loss_mag > 1000
            if torch.any(mask):
                print("[Found extrem epe. Filtered out. Max is {}. Ratio is {}]".format(torch.max(loss_mag), torch.mean(mask.float())))
                _valid = _valid & (~mask[:, :, None])

        flow_loss += i_weight * (_valid * i_loss).mean()
    
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=2).sqrt()
    
    if valid.shape[1]== 1:
        valid = valid.repeat(1,2,1,1)
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, NAN_flag