import torch
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from typing import cast
import sys 
import os
sys.path.append(os.path.abspath("../monarch/dnam2/bert/"))

from main import build_model
import src.create_bert as bert_module
import src.create_model as model_module
from transformers import AutoTokenizer, Trainer, TrainingArguments

def load_model(model_path = "../monarch/dnam2/bert/slurm/composer/local-bert-checkpoints/lrcorrect__monarch-mixer-pretrain-786dim-80m-parameters/ep39-ba200000-rank0.pt"):
    MODEL_PATH = model_path
    state_dict = torch.load(MODEL_PATH)
    state_dict = state_dict["state"]["model"]
    state_dict_without_prefix = {k.replace('model.', ''): v for k, v in state_dict.items()}





    yaml_path = "../monarch/dnam2/bert/yamls/pretrain/micro_dna_monarch-mixer-pretrain-786dim-80m-parameters.yaml"

    tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")


    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)
    print(cfg.max_duration)
    model = model_module.create_model(cfg.model.get("model_config"))

    model.load_state_dict(state_dict_without_prefix)
    return model