from omegaconf import DictConfig
from .COMTF import ComBrainTF


def model_factory(config: DictConfig):
    if config.model.name != "ComBrainTF":
        raise ValueError("This package contains only the CenFormer/ComBrainTF backbone.")
    return ComBrainTF(config).cuda()
