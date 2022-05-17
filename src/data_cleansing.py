
import json
import os
from cProfile import label
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

# dir path
ROOT_DIR = Path.cwd().parent.resolve()
DATA_DIR = ROOT_DIR / "dataset"
TEST_IMAGE_DIR = DATA_DIR / "test_images"
TRAIN_IMAGE_DIR = DATA_DIR / "train_images"

# fix typo map c.f. https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/305574
FIX_NAME_MAPPING = {"bottlenose_dolpin": "bottlenose_dolphin",
                    "kiler_whale": "killer_whale",
                    "pilot_whale": "short_finned_pilot_whale",
                    "globis": "short_finned_pilot_whale",
                    "beluga": "beluga_whale"}

print("\n... TRAIN DATAFRAME ...\n")
train_df = pd.read_csv(DATA_DIR/"train.csv")
ss_df = pd.read_csv(DATA_DIR/"sample_submission.csv")
# add meta image data
train_df["img_path"] = os.path.join(
    DATA_DIR, "train_images")+"/"+train_df.image
# fix name
train_df["species"] = train_df["species"].apply(
    lambda x: FIX_NAME_MAPPING[x] if x in FIX_NAME_MAPPING.keys() else x)
# 個体数が何頭イルカ
train_df["n_img_of_ind"] = train_df.individual_id.map(
    train_df.individual_id.value_counts().to_dict())

# species encoding
all_species = sorted(train_df.species.unique().tolist())
species_int2str_lbl_map = {i: _s for i, _s in enumerate(all_species)}
species_str2int_lbl_map = {v: k for k, v in species_int2str_lbl_map.items()}
with open(DATA_DIR/"species_label.json", "w") as f:
    json.dump(species_str2int_lbl_map, f)
pprint(species_str2int_lbl_map)
N_SPECIES = len(all_species)
print(f"\n unique species..{N_SPECIES} \n")

# inidiviasuals encoding
all_individuals = sorted(train_df.individual_id.unique().tolist())
ind_int2str_lbl_map = {i: _s for i, _s in enumerate(all_individuals)}
ind_str2int_lbl_map = {v: k for k, v in ind_int2str_lbl_map.items()}
with open(DATA_DIR/"indivisual_label.json", "w") as f:
    json.dump(ind_str2int_lbl_map, f)
N_INDIVIDUALS = len(all_individuals)
print(f"\n unique species..{N_INDIVIDUALS} \n")

train_df["ind_sparse_lbl"] = train_df["individual_id"].map(ind_str2int_lbl_map)
train_df["species_sparse_lbl"] = train_df["species"].map(
    species_str2int_lbl_map)

train_df.to_csv(DATA_DIR / "train_cleansing.csv", index=False)
print(train_df.head())
