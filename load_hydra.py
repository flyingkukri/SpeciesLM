import torch
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from typing import cast
import sys 
import os
#sys.path.append(os.path.abspath(".."))


sys.path.append(os.path.abspath("../hydra/hydra/bert/src/"))
import create_hydra_model as model_module_hydra
from transformers import AutoTokenizer, Trainer, TrainingArguments


def load_hydra_model(model_path = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/slurm/local-bert-checkpoints/hydra_embed/ep18-ba91000-rank0.pt",
            yaml_path = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/yamls/pretrain/hydra_noembed.yaml"):
    state_dict = torch.load(model_path)
    state_dict = state_dict["state"]["model"]
    state_dict_without_prefix = {k.replace('model.', ''): v for k, v in state_dict.items()}

    tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")
    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)


    print("loading hydra")
    model = model_module_hydra.create_hydra_model(cfg.model.get("model_config"))

    model.load_state_dict(state_dict_without_prefix)
    return model

