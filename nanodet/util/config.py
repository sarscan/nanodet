from .yacs import CfgNode
# 这个new allowed 很好理解，就是这里有些默认的配置，而配置文件里还有更多的配置，就是允许合并配置文件里的新配置
cfg = CfgNode(new_allowed=True)
cfg.save_dir = "./"
# common params for NETWORK
cfg.model = CfgNode(new_allowed=True)
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.fpn = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
cfg.device.precision = 32
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)
