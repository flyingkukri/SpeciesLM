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

def load_m2_model(model_path = None,
            yaml_path = "../dnam2/bert/yamls/pretrain/micro_dna_monarch-mixer-pretrain-786dim-80m-parameters.yaml"):
    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)
    sys.path.append(os.path.abspath("../dnam2/bert/"))
    import src.create_m2_model as model_module_m2
    print("loading m2")
    model = model_module_m2.create_m2_model(cfg.model.get("model_config"))
    sys.path.remove(os.path.abspath("../dnam2/bert/"))
    
    if model_path is None:
        print("Not initialising weights")
    else:
        state_dict = torch.load(model_path)
        state_dict = state_dict["state"]["model"]
        state_dict_without_prefix = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict_without_prefix)
    return model
    