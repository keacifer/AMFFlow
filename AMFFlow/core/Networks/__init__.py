def build_network(cfg):
    name = cfg.network
    if name == 'AMFFlow':
        from .AMFFlow.AMFFlowNet import AMFFlowNet as network
    
    else:
        raise ValueError(f"Network = {name} is not a valid name!")

    return network(cfg[name])
