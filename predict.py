import math
import itertools
import collections
from collections.abc import Mapping
import numpy as np
import pandas as pd
import tqdm
import os
import torch

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

from datasets import Dataset

import sys
import os
sys.path.append(os.path.abspath("../hydra/hydra/bert"))
from collate import DataCollatorForLanguageModelingSpan

def chunkstring(string, length):
    # chunks a string into segments of length
    return (string[0+i:length+i] for i in range(0, len(string), length))

def kmers(seq, k=6):
    # splits a sequence into non-overlappnig k-mers
    return [seq[i:i + k] for i in range(0, len(seq), k) if i + k <= len(seq)]

def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]

def one_hot_encode(gts, dim=5):
    # one-hot encodes the sequence
    result = []
    nuc_dict = {"A":0,"C":1,"G":2,"T":3}
    for nt in gts:
        vec = np.zeros(dim)
        vec[nuc_dict[nt]] = 1
        result.append(vec)
    return np.stack(result, axis=0)

def class_label_gts(gts):
    # make labels into ground truths
    nuc_dict = {"A":0,"C":1,"G":2,"T":3}
    return np.array([nuc_dict[x] for x in gts])

def tok_func_standard(x, seq_col): return tokenizer(" ".join(kmers_stride1(x[seq_col])))

def tok_func_species(x, species_proxy, seq_col):
    res = tokenizer(species_proxy + " " +  " ".join(kmers_stride1(x[seq_col])))
    return res

def tok_func_embed(x, species_proxy, seq_col):
    start_species_id = tokenizer.convert_tokens_to_ids(["GGGGGG"])[0]
    res = tokenizer(" ".join(kmers_stride1(x[seq_col])))
    species_id = tokenizer.convert_tokens_to_ids(species_proxy)
    species_id = species_id - start_species_id
    res["species_id"] = np.array([species_id])
    return res

def count_special_tokens(tokens, tokenizer, where = "left"):
    count = 0
    if where == "right":
        tokens = tokens[::-1]
    for pos in range(len(tokens)):
        tok = tokens[pos]
        if tok in tokenizer.all_special_ids:
            count += 1
        else:
            break
    return count

#TODO change path
scer_path = "/s/project/semi_supervised_multispecies/Downstream/Sequences/Annotation/Sequences/saccharomyces_cerevisiae/saccharomyces_cerevisiae_three_prime.parquet"

# Downstream
scer_ds_path = "/s/project/semi_supervised_multispecies/Zenodo/zenodo/data/Sequences/Annotation/Sequences/saccharomyces_cerevisiae/saccharomyces_cerevisiae_three_prime.parquet"
pombe_ds_path = "/s/project/semi_supervised_multispecies/Zenodo/zenodo/data/Sequences/Annotation/Sequences/schizosaccharomyces_pombe/schizosaccharomyces_pombe_three_prime_remapped.tsv"
pombe_pq_path = "/s/project/semi_supervised_multispecies/Zenodo/zenodo/data/Sequences/Annotation/Sequences/schizosaccharomyces_pombe/schizosaccharomyces_pombe_three_prime.parquet"
three_sequences = "/s/project/semi_supervised_multispecies/Downstream/Sequences/Annotation/Sequences/schizosaccharomyces_pombe/schizosaccharomyces_pombe_three_prime_remapped.tsv"

mpra_path = "/s/project/semi_supervised_multispecies/Zenodo/zenodo/data/Downstream_Targets/segal_2015.tsv"

seq_df_path = scer_ds_path

seq_col = "three_prime_seq" # name of the column in the df that stores the sequences
#seq_col = "three_prime_region" # for s. pombe

kmer_size = 6 # size of kmers, always 6
proxy_species = "candida_glabrata" # species token to use
# proxy_species = "schizosaccharomyces_pombe"
pred_batch_size = 128*3 # batch size for rolling masking
target_layer = (8,) # what hidden layers to use for embedding
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")
from load_hydra import load_hydra_model
#from load_m2 import load_m2_model

monarch_100k_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/composer/local-bert-checkpoints/lrcorrect__monarch-mixer-pretrain-786dim-80m-parameters/ep19-ba100000-rank0.pt"
monarch_200k_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/composer/local-bert-checkpoints/lrcorrect__monarch-mixer-pretrain-786dim-80m-parameters/ep39-ba200000-rank0.pt"
agnostic_monarch_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/composer/local-bert-checkpoints/nonspecies_monarch-mixer-pretrain-786dim-80m-parameters/ep19-ba100000-rank0.pt"
species_embed_100k_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/species_embedding/local-bert-checkpoints/embed_full_dna_monarch-mixer-pretrain-786dim-80m-parameters2/ep19-ba100000-rank0.pt"
species_embed_200k_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/species_embedding/local-bert-checkpoints/embed_full_dna_monarch-mixer-pretrain-786dim-80m-parameters2/ep39-ba200000-rank0.pt"
species_embed_yaml = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/yamls/pretrain/embed_dna_monarch-mixer-pretrain-786dim-80m-parameters.yaml"
species_embed_90k_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/species_embedding/local-bert-checkpoints/embed_full_dna_monarch-mixer-pretrain-786dim-80m-parameters2/ep18-ba91000-rank0.pt"
species_embed_larger_yaml = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/yamls/pretrain/embed_large_dna_monarch-mixer-pretrain-786dim-80m-parameters.yaml"
species_embed_37k_path = "/data/nasif12/home_if12/huan/monarch/dnam2/bert/slurm/species_embedding/local-bert-checkpoints/embed_full_dna_monarch-mixer-pretrain-786dim-80m-parameters2/ep7-ba37000-rank0.pt"

hydra_embed_yaml = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/yamls/pretrain/hydra.yaml"
hydra_noembed_yaml = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/yamls/pretrain/hydra_noembed.yaml"
hydra_91k_path = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/slurm/local-bert-checkpoints/hydra_embed/ep18-ba91000-rank0.pt"
hydra_37k_path = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/slurm/local-bert-checkpoints/hydra_embed_correct/ep7-ba37000-rank0.pt"
hydra_100k_path = "/data/nasif12/home_if12/huan/monarch/hydra/hydra/bert/slurm/local-bert-checkpoints/hydra_embed_correct/ep19-ba100000-rank0.pt"

model = load_hydra_model(model_path = hydra_100k_path,
                    yaml_path = hydra_embed_yaml)

#model = load_m2_model(model_path = species_embed_37k_path,
#                 yaml_path = species_embed_yaml)

# model = AutoModelForMaskedLM.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")

device = "cuda"

model.to(torch.bfloat16).to(device)
model.to(torch.float16).to(device)
model.to(device)
model.eval()

print("Done")

dataset = pd.read_parquet(seq_df_path)
# For reading the pombe dataset
#dataset = pd.read_csv(seq_df_path, sep="\t")
dataset[seq_col] = dataset[seq_col].str[:300] # truncate longer sequences
dataset = dataset.loc[dataset[seq_col].str.len() == 300] # throw out too short sequences

# delete the species token for the agnostic model
tok_func = lambda x: tok_func_embed(x, proxy_species,seq_col)
# tok_func = lambda x: tok_func_standard(x,seq_col)

ds = Dataset.from_pandas(dataset[[seq_col]])

tok_ds = ds.map(tok_func, batched=False,  num_proc=2)

rem_tok_ds = tok_ds.remove_columns(seq_col)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
data_loader = torch.utils.data.DataLoader(rem_tok_ds, batch_size=1, collate_fn=data_collator, shuffle = False)

def predict_on_batch_generator(tokenized_data, dataset, seq_idx,
                               special_token_offset,
                               kmer_size = kmer_size,
                               seq_col = seq_col,
                               pred_batch_size = pred_batch_size):
    model_input_unaltered = tokenized_data['input_ids'].clone()
    label = dataset.iloc[seq_idx][seq_col]
    label_len = len(label)
    if label_len < kmer_size:
        print("This should not occur")
        return torch.zeros(label_len,label_len,5)
    else:
        diag_matrix = torch.eye(tokenized_data['input_ids'].shape[1]).numpy()
        masked_indices = np.apply_along_axis(lambda m : np.convolve(m, [1] * 6, mode = 'same' ),axis = 1, arr = diag_matrix).astype(bool)
        masked_indices = torch.from_numpy(masked_indices)
        masked_indices = masked_indices[2+special_token_offset:label_len-(kmer_size-1)-3+special_token_offset]
        res = tokenized_data['input_ids'].expand(masked_indices.shape[0],-1).clone()
        res[masked_indices] = 4
        yield res.shape[0] # provide the total size
        for batch_idx in range(math.ceil(res.shape[0]/pred_batch_size)):
            res_batch = res[batch_idx*pred_batch_size:(batch_idx+1)*pred_batch_size]
            res_batch = res_batch.to(device)

            species_id = tokenized_data['species_id']
            species_id = species_id.expand(res_batch.shape).clone()
            species_id = species_id.to(device)
            with torch.no_grad():
                computation = model(res_batch, species_id=species_id)
                logits = computation["logits"].detach()
                #if "logits" in computation:
                #    logits = computation["logits"].detach()
                #else:
                #    logits = computation["prediction_logits"].float().detach()
                fin_calculation = logits
            yield fin_calculation, res


# make a convolutional filter for each nt
# the way this works:
# The kmer ACGTGC
# maps to token 739
# the last nt is C
# this would be the prediction for the masked nucleotide
# from this kmer, if the kmer is the first in masked span
# so the first row of column 739 searches for C
# in other words filter_ijk = 1 for i = 0, j = 739, k = 2
vocab = tokenizer.get_vocab()
kmer_list = ["".join(x) for x in itertools.product("ACGT",repeat=6)]
nt_mapping = {"A":0,"C":1,"G":2,"T":3}
prb_filter = np.zeros((kmer_size, 4**kmer_size, 4))
for kmer in kmer_list:
    token = vocab[kmer] - 5 # there are 5 special tokens
    for idx, nt in enumerate(kmer):
        nt_idx = nt_mapping[nt]
        prb_filter[5-idx, token, nt_idx] = 1
prb_filter = torch.from_numpy(prb_filter)
prb_filter = prb_filter.to(device)

def extract_prbs_from_pred(kmer_prediction,
                           label_pos,
                           max_pos,
                           prb_filter=prb_filter,
                           kmer_size=kmer_size):
    # label_pos = position of actual nucleotide in sequence
    nt_preds = kmer_prediction[label_pos:(label_pos+kmer_size),:] # extract the right kmers
    nt_preds = nt_preds.unsqueeze(2).expand((nt_preds.shape[0],nt_preds.shape[1],4)) # repeat along nt dimension
    nt_preds = (nt_preds*prb_filter).sum(axis=1) # filter and add over tokens
    nt_preds = nt_preds.sum(axis=0)
    nt_prbs = nt_preds/nt_preds.sum() # renormalize
    return nt_prbs.cpu().numpy()

predicted_prbs,gts = [],[]
prev_len = 0

for no_of_index, tokenized_data in tqdm.tqdm(enumerate(data_loader)):
    print(no_of_index)
    #if no_of_index > 10:
    #    break
    label = dataset.iloc[no_of_index][seq_col]
    label_len = len(label)

    left_special_tokens = count_special_tokens(tokenized_data['input_ids'].numpy()[0], tokenizer, where="left")
    right_special_tokens = count_special_tokens(tokenized_data['input_ids'].numpy()[0], tokenizer, where="right")

    # Edge case: for a sequence less then 11 nt
    # we cannot even feed 6 mask tokens
    # so we might as well predict random
    if label_len < 11:
        #print (no_of_index)
        for i in range(label_len):
            predicted_prbs.append(torch.tensor([0.25,0.25,0.25,0.25]))
            gts.append(label[i])
        added_len = len(predicted_prbs) - prev_len
        prev_len = len(predicted_prbs)
        assert added_len == len(label)
        continue

    # we do a batched predict to process the sequence
    batch_start = 0
    pos = 0
    prediction_generator = predict_on_batch_generator(tokenized_data, dataset, no_of_index, special_token_offset = left_special_tokens)
    max_idx = next(prediction_generator)
    for predictions, res in prediction_generator:

        # prepare predictions for processing
        logits = predictions[:,:,5:(5+prb_filter.shape[1])] # remove any non k-mer dims
        kmer_preds = torch.softmax(logits,dim=2)
        # remove special tokens:
        kmer_preds = kmer_preds[:,(left_special_tokens):(kmer_preds.shape[1] - right_special_tokens),:]
        max_pos = kmer_preds.shape[1] - 1
        # pad to predict first 5 and last 5 nt
        padded_tensor = torch.zeros((kmer_preds.shape[0],2*(kmer_size-1) + kmer_preds.shape[1],kmer_preds.shape[2]),device=device)
        padded_tensor[:,kmer_size-1:-(kmer_size-1),:] = kmer_preds
        kmer_preds = padded_tensor

        while pos < label_len:
            # get prediction
            theoretical_idx = min(max(pos-5,0),max_idx-1) # idx if we did it all in one batch
            actual_idx = max(theoretical_idx - batch_start,0)
            if actual_idx >= kmer_preds.shape[0]:
                break
            kmer_prediction = kmer_preds[actual_idx]
            nt_prbs = extract_prbs_from_pred(kmer_prediction=kmer_prediction,
                                             label_pos=pos,
                                             max_pos=max_pos)
            predicted_prbs.append(nt_prbs)
            #print(nt_prbs.shape)
            #print(label_len)
            # extract ground truth
            gt = label[pos]
            gts.append(gt)
            # update
            pos += 1

        batch_start = pos - 5

    added_len = len(predicted_prbs) - prev_len
    prev_len = len(predicted_prbs)
    assert added_len == len(label)


prbs_arr = np.stack(predicted_prbs).reshape((no_of_index + 1, 300, 4))
prbs_tensor = torch.Tensor(np.stack(predicted_prbs))
torch.save(prbs_tensor, 'hydra100k_pred.pt')
