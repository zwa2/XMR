#!/bin/bash

model_dir="./checkpoints"
mkdir -p $model_dir

python -u ./prune_xmr.py \
-onto ./pathways/DNA_Repair.txt \
-gene2id ./data/gene2ind.txt \
-cell2id ./data/cell2ind.txt \
-drug2id ./data/drug2ind.txt \
-cellline ./data/cell2mutation.txt \
-cancer_type TNBC \
-modeldir $model_dir
