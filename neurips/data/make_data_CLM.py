#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import gc
import copy
import yaml
import pickle
import random
import joblib
import shutil
from time import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import average_precision_score as APS
import duckdb


import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
from torch.nn import BCELoss


import timm
from mamba_ssm import Mamba
from transformers import AutoModel, AutoTokenizer

import albumentations as A
from albumentations.pytorch import ToTensorV2


# use one device only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = duckdb.connect()


# In[2]:


class CFG:
    NUM = 20000
    TEST_No = 3
    TEST_NUM = int(1674896/16 * TEST_No)
    TEST_OFFSET = int(TEST_NUM * (TEST_No-1))
    LR = 0.001
    WD = 1e-4
    NBR_FOLDS = 5
    SELECTED_FOLDS = [0, 1, 2, 3, 4]
    TRAIN_ENC_PATH = Path('../../data/external/train_enc.parquet')
    TEST_ENC_PATH = Path('../../data/external/test_enc.parquet')
    TRAIN_PATH = Path('../../data/raw/train.parquet')
    TEST_PATH = Path('../../data/raw/test.parquet')
    OUTPUT_PATH = Path(f'../../data/processed/{NUM}_50per_CLM.parquet')
    TEST_OUTPUT_PATH = Path(f'../../data/processed/test_CLM_{TEST_OFFSET}_to_{TEST_NUM}.parquet')
    BATCH_SIZE = 128
    EPOCHS = 5
    PATIENCE = 5
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.5
    is_test = True


# In[3]:


if not CFG.is_test:
    train = con.query(f"""(SELECT *
                            FROM parquet_scan('{CFG.TRAIN_PATH}')
                            WHERE binds = 0
                            ORDER BY random()
                            LIMIT {int(CFG.NUM/2)})
                            UNION ALL
                            (SELECT *
                            FROM parquet_scan('{CFG.TRAIN_PATH}')
                            WHERE binds = 1
                            ORDER BY random()
                            LIMIT {int(CFG.NUM/2)})""").df()
else:
    test = con.query(f"""(SELECT *
                        FROM parquet_scan('{CFG.TEST_PATH}')
                        LIMIT {CFG.TEST_NUM}
                        OFFSET {CFG.TEST_OFFSET}
                        )""").df()



# In[4]:


if not CFG.is_test:
    print(train.head())
    print(train.tail())
else:
    print(test.head())
    print(test.tail())


# In[5]:


if not CFG.is_test:
    smiles = train['molecule_smiles']#.unique()
    print(len(smiles))
else:
    smiles = test['molecule_smiles']#.unique()
    print(len(smiles))


# In[6]:


# 104681 rows take about 10 minutes.
# load pre-trained ChemBERTa model checkpoint and tokenizer
cb_tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-10M-MLM')
cb_model = AutoModel.from_pretrained('DeepChem/ChemBERTa-10M-MLM')
cb_model.eval()

# tokenize SMILES
cb_encoded_inputs = cb_tokenizer(list(smiles), padding=True, truncation=True, return_tensors="pt")

# calculate embeddings
with torch.no_grad():
    outputs = cb_model(**cb_encoded_inputs)

# extract pooled output
cb_embeddings = outputs.pooler_output

cb_embeddings_df = pd.DataFrame(cb_embeddings.numpy())
cb_embeddings_df.head()


# In[ ]:


# df_repeated = cb_embeddings_df.loc[cb_embeddings_df.index.repeat(3)].reset_index(drop=True)


# In[ ]:


if not CFG.is_test:
    cb_embeddings_df = pd.concat([train['id'], cb_embeddings_df], axis=1)
    binds = train[['binds', 'protein_name']]
    binds['bind1'] = train.apply(lambda row: row['binds'] if row['protein_name'] == 'BRD4' else 0, axis=1)
    binds['bind2'] = train.apply(lambda row: row['binds'] if row['protein_name'] == 'HSA' else 0, axis=1)
    binds['bind3'] = train.apply(lambda row: row['binds'] if row['protein_name'] == 'sEH' else 0, axis=1)
    cb_embeddings_df = pd.concat([cb_embeddings_df, binds], axis=1)
else:
    cb_embeddings_df = pd.concat([test['id'], cb_embeddings_df], axis=1)
    cb_embeddings_df = pd.concat([cb_embeddings_df, test['protein_name']], axis=1)


# In[ ]:


print(cb_embeddings_df.head())
print(cb_embeddings_df.tail())


# In[ ]:


cb_embeddings_df.columns = cb_embeddings_df.columns.astype(str)

if not CFG.is_test:
    cb_embeddings_df.to_parquet(CFG.OUTPUT_PATH)
else:
    cb_embeddings_df.to_parquet(CFG.TEST_OUTPUT_PATH)


# In[ ]:





# In[ ]:





# In[ ]:




