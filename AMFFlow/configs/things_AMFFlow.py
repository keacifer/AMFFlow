from yacs.config import CfgNode as CN
_CN = CN()
_CN.name = ''
_CN.suffix = 'AMFlownet'
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 10000
_CN.image_size = [400, 720]
_CN.use_smoothl1 = False
_CN.filter_epe = False
_CN.critical_params = []
_CN.network = 'AMFFlow'
_CN.restore_steps = 0
_CN.mixed_precision = False
_CN.val_decoder_depth = 12
_CN.gamma = 0.85
_CN.input_frames = 3
### Train
_CN.freeze_module = False
_CN.load_strict = False
_CN.add_noise = True
### MODEL
_CN.AMFFlow = CN()
_CN.AMFFlow.pretrain = True
_CN.AMFFlow.cnet = 'twins'
_CN.AMFFlow.fnet = 'twins'
_CN.AMFFlow.gma = 'GMA-SK2'
_CN.AMFFlow.down_ratio = 8
_CN.AMFFlow.feat_dim = 256
_CN.AMFFlow.corr_fn = 'default'
_CN.AMFFlow.corr_levels = 4
_CN.AMFFlow.corr_radius = 4
_CN.AMFFlow.decoder_depth = 12



### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 5e-5

_CN.trainer.split_train = True
_CN.trainer.pretrained_lr = 5e-5
_CN.trainer.new_lr = 5e-5

_CN.trainer.pct_start = 0.2
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 20000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
