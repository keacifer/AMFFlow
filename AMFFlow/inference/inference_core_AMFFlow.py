import torch


class InferenceCore:
    def __init__(self, network, config):       
        self.model = network
        self.config = config
    def step(self, images, flow_init=None):
        
        flow_pre = self.model(images)
        return  flow_pre
