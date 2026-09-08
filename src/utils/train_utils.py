from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple



def parse_configs(config_path: str) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    config = OmegaConf.load(config_path)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    gmm_config = config.get("gmm", None)
    fid_config = config.get("fid", None)
    data_limit_config = config.get("data_limit", None)
    data_config = config.get("data", None)  # WebDataset or other data configs
    vae_prior_config = config.get("vae_prior", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, gmm_config, fid_config, data_limit_config, data_config, vae_prior_config

def none_or_str(value):
    if value == 'None':
        return None
    return value