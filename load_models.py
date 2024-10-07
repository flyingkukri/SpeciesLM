import torch
from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from typing import cast
import sys 
import os
#sys.path.append(os.path.abspath(".."))


#from dnam2.main import build_model
#import dnam2.bert.src.create_bert as bert_module
from transformers import AutoTokenizer, Trainer, TrainingArguments

sys.path.append(os.path.abspath("../hydra/hydra/bert/src/create_hydra_model"))
from create_hydra_model import create_hydra_model

sys.path.append(os.path.abspath("../dnam2/bert/src/create_hydra_model"))
from create_m2_model import create_m2_model


def load_hydra(model_path, yaml_path):
    return load_model(create_hydra_model)

def load_m2(model_path, yaml_path):
    return load_model(create_m2_model)

def load_model(model_creation_function, 
            model_path = None,
            yaml_path = None):
    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)
    sys.path.append(os.path.abspath("../dnam2/bert/"))
    import src.create_model as model_module_m2
    print("loading m2")
    model = model_module_m2.create_model(cfg.model.get("model_config"))
    sys.path.remove(os.path.abspath("../dnam2/bert/"))
    
    if model_path is None:
        print("Not initialising weights")
    else:
        state_dict = torch.load(model_path)
        state_dict = state_dict["state"]["model"]
        state_dict_without_prefix = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict_without_prefix)
    return model