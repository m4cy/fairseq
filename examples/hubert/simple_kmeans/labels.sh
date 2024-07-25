#!/bin/bash

pwd

# ckpt_path="/home/macyhuang/hubert_training/hubert_base_ls960.pt"
# models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
# model = models[0]

km_train_path="/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/train.km"
km_valid_path="/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/valid.km"

# CUDA_VISIBLE_DEVICES=0,1 python dump_hubert_feature.py /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir/ valid /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/None/checkpoints/checkpoint_last.pt 6 1 0 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/feat_dir_hubert
# CUDA_VISIBLE_DEVICES=0,1 python learn_kmeans.py /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/feat_dir_hubert valid 1 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_hubert/valid.km 500 --percent 0.1
# CUDA_VISIBLE_DEVICES=0,1 python dump_km_label.py /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/feat_dir_hubert valid /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_hubert/valid.km 1 0 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_hubert
# for rank in {0..28}
# do
    # CUDA_VISIBLE_DEVICES=1,2,3 python3 dump_mfcc_feature.py tsv_dir train 29 $rank feat_dir
    # CUDA_VISIBLE_DEVICES=1,2,3 python3 dump_mfcc_feature.py tsv_dir valid 29 $rank feat_dir

#         # CUDA_VISIBLE_DEVICES=1,2,3 python dump_hubert_feature.py tsv_dir split $ckpt_path $layer 29 $rank feat_dir
#     # CUDA_VISIBLE_DEVICES=1,2,3 python learn_kmeans.py feat_dir train 29 $km_train_path 100 --percent 0.1 --max_no_improvement 40000
# # 2022-09-28 14:53:13 | INFO | learn_kmeans | total intertia: 1546.52087
# # 2022-09-28 14:53:13 | INFO | learn_kmeans | finished successfully
#     # CUDA_VISIBLE_DEVICES=1,2,3 python learn_kmeans.py feat_dir valid 29 $km_valid_path 100 --percent 0.1 --max_no_improvement 40000
# #     2022-09-28 14:53:52 | INFO | learn_kmeans | total intertia: 1445.88800
# # 2022-09-28 14:53:52 | INFO | learn_kmeans | finished successfully
#     # CUDA_VISIBLE_DEVICES=1,2,3 python dump_km_label.py feat_dir train $km_train_path 29 $rank lab_dir
#     # CUDA_VISIBLE_DEVICES=1,2,3 python dump_km_label.py feat_dir valid $km_valid_path 29 $rank lab_dir
#     #     cat lab_dir/valid_"$rank"_29.km
#     # done > lab_dir/valid.km
# done
# sed 's/^.\{,25\}//' splits.tsv > split.tsv
# /home/macyhuang/hubert_training/LibriSpeech/test-clean	

# while read line; do
#     FILENAME=/home/heiko/dummy/packages.txt
#     FILESIZE=$(stat -c%s '$FILENAME')
#     echo 'Size of $FILENAME = $FILESIZE bytes.'
#     # FILESIZE=$(stat -c%s "$line")
#     # echo "$line"
#     # echo "$FILESIZE"
# done < split.tsv
# lab_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_hubert
# split=train
# rank=0
# nshard=1
# CUDA_VISIBLE_DEVICES=7 python dump_hubert_feature.py /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir 'train' /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/None/checkpoints/checkpoint_last.pt 6 2 0 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/feat_dir_hubert
# CUDA_VISIBLE_DEVICES=7 python dump_hubert_feature.py /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir 'train' /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/None/checkpoints/checkpoint_last.pt 6 2 1 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/feat_dir_hubert

# CUDA_VISIBLE_DEVICES=7 python learn_kmeans.py ./train-clean-100 'train' 1 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/km_path 500 --percent 0.1
# CUDA_VISIBLE_DEVICES=7 python dump_km_label.py /home/macyhuang/train-clean-100 'train' /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/km_path 2 0 /home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_hubert
# for rank in $(seq 0 $((nshard - 1))); do
#   cat $lab_dir/${split}_${rank}_${nshard}.km
# done > $lab_dir/${split}.km

# for x in $(seq 0 $((n_clusters - 1))); do
#   echo "$x 1"
# done >> $lab_dir/dict.km.txt
CUDA_VISIBLE_DEVICES=2,3 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
  --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
  task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir task.labels='["km"]' model.label_rate=100

# # figuring out how to subclass hubert
CUDA_VISIBLE_DEVICES=5,6 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
  --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
  task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir task.labels='["km"]' model.label_rate=50
# import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

# this is the right one
# CUDA_VISIBLE_DEVICES=5,6 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
#   --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
#   --config-name hubert_spec_base_librispeech \
#   task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
#   task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_hubert task.labels='["km"]' model.label_rate=100 \
  
# september 18/october 23/february 1
# CUDA_VISIBLE_DEVICES=5,6 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
#   --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
#   --config-name spectrohubert_base_librispeech \
#   task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
#   task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_spectro_hubert task.labels='["km"]' model.label_rate=100

# CUDA_VISIBLE_DEVICES=5,6 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
#   --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
#   --config-name spectrohubert_base_librispeech \
#   task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
#   task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_spectro_hubert task.labels='["km"]' model.label_rate=100


CUDA_VISIBLE_DEVICES=2,3 python fairseq_cli/hydra_train.py \
    --config-dir spectrohubert/config/pretrain \
    --config-name spectrohubert_base_librispeech.yaml \
    common.user_dir=/home/macyhuang/hubert_training/fairseq/spectrohubert \
    task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
    task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_spectro_hubert task.labels='["km"]' model.label_rate=100

# CUDA_VISIBLE_DEVICES=6 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
#   --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
#   --config-name spectrohubert_base_librispeech \
#   task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
#   task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir_spectro_hubert task.labels='["km"]' model.label_rate=50 \
  

# CUDA_VISIBLE_DEVICES=0 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
#   --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/finetune \
#   --config-name base_spec \
  # task.data= \
  # task.label_dir= \
  # model.w2v_path=
# for x in {0..100}; do
#   echo "$x 1"
# done >> lab_dir/dict.km.txt

# CUDA_VISIBLE_DEVICES=0 python ~/hubert_training/fairseq/fairseq_cli/hydra_train.py \
#   --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/finetune \
#   --config-name base_10h \
  # task.data=  \
  # task.label_dir= 

# october 23 re-clustering
# CUDA_VISIBLE_DEVICES=2 python dump_hubert_feature.py tsv_dir "valid" None/checkpoints/checkpoint_last.pt 12 1 0 feat_dir_spectro_hubert
# CUDA_VISIBLE_DEVICES=2 python learn_kmeans.py feat_dir_spectro_hubert "valid" 1 train_spectro_hubert 500 --percent 0.1
# CUDA_VISIBLE_DEVICES=2 python dump_km_label.py feat_dir_spectro_hubert "valid" train_spectro_hubert 1 0 lab_dir_spectro_hubert
# cat lab_dir_spectro_hubert/valid_0_1.km > lab_dir_spectro_hubert/valid.km
# for x in $(seq 0 499); do
#   echo "$x 1"
# done >> lab_dir_spectro_hubert/dict.km.txt

# february 8
CUDA_VISIBLE_DEVICES=0,2,3  python /home/macyhuang/hubert_training/fairseq/fairseq_cli/hydra_train.py \
    --config-dir /home/macyhuang/hubert_training/fairseq/fairseq/models/spectrohubert \
    --config-name spectrohubert_base_librispeech \
    common.user_dir=/home/macyhuang/hubert_training/fairseq/fairseq/models/spectrohubert \
    task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
        task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir task.labels='["km"]' model.label_rate=50

CUDA_VISIBLE_DEVICES=1,4 python /home/macyhuang/hubert_training/fairseq/fairseq_cli/hydra_train.py \
    --config-dir /home/macyhuang/hubert_training/fairseq/examples/hubert/config/pretrain \
    --config-name hubert_base_librispeech \
    common.user_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert \
    task.data=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/tsv_dir \
        task.label_dir=/home/macyhuang/hubert_training/fairseq/examples/hubert/simple_kmeans/lab_dir task.labels='["km"]' model.label_rate=50
