# fairseq_mmt/fairseq/data/image_dataset.py to set feat dataloader

# codes have been edited:
# fairseq_mmt/fairseq/data/image_dataset.py
# fairseq_mmt/fairseq/tasks/image_multimodal_translation.py
# fairseq_mmt/fairseq/data/image_language_pair_dataset.py


# set before using:
# device, image_feat, max_tokens, image_dataset.py->random.shuffle

#! /usr/bin/bash
set -e

# device=0,1,2,3,4,5,6,7
# device=0,1,2,3
device=4,5
# device=0
task=opus-zh2en
# image_feat=videoMAE
image_feat=nmt_4
patience=20
keep_last_epochs=10

mask_data=mask0
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'opus-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opus.ja-en
elif [ $task == 'opus-random-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opus-random.ja-en
elif [ $task == 'VISA-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=VISA.ja-en
elif [ $task == 'transet-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=transet.ja-en
elif [ $task == 'opus-zh2en' ]; then
	src_lang=zh
	tgt_lang=en
	data_dir=opus.zh-en
fi

criterion=label_smoothed_cross_entropy
fp16=0 #0
lr=0.005
warmup=2000
max_tokens=4000
# max_tokens=2000 # for videoMAE and i3d
# max_tokens=8000 # for c4c
update_freq=1
max_update=800000
dropout=0.3

arch=transformer_top
# SA_attention_dropout=0.1
# SA_image_dropout=0.1

if [ $image_feat == "vit_tiny_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=192
elif [ $image_feat == "vit_small_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=384
elif [ $image_feat == "vit_base_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=768
elif [ $image_feat == "vit_large_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=1024
elif [ $image_feat == "i3d" ] || [ $image_feat == "i3d1" ]|| [ $image_feat == "i3d_random" ]|| [ $image_feat == "i3d_random1" ]; then
	image_feat_path=/dataset/OpusEJ/OpusEJ_i3d_feature
	image_feat_dim=2048 # (32, 2048)
	image_feat_whole_dim="32 2048" # (32, 2048)
elif [ $image_feat == "c4c" ] || [ $image_feat == "c4c_random" ]; then
	image_feat_path=/dataset/OpusEJ/OpusEJ_c4c_feature
	image_feat_dim=512
	image_feat_whole_dim="12 512" # (12, 512)
elif [ $image_feat == "videoMAE" ] || [ $image_feat == "videoMAE_random" ]; then
	image_feat_path=/dataset/OpusEJ/OpusEJ_videoMAE_feature_224
	image_feat_dim=384
	image_feat_whole_dim="1568 384" # (1568, 384)
elif [ $image_feat == "nmt" ]; then
	image_feat_path=/
	image_feat_dim=0
	image_feat_whole_dim="0 0" # (1568, 384)
fi

# multi-feature
#image_feat_path=data/vit_large_patch16_384 data/vit_tiny_patch16_384
#image_feat_dim=1024 192

cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

# export PYTHONPATH=$PYTHONPATH:/home/code/fairseq_mmt/fairseq
# echo $PYTHONPATH
#   --user-dir fairseq/tasks
#   --share-all-embeddings
#   --num-workers 2
#   --find-unused-parameters
#   --model-parallel-size
#   --load-checkpoint-on-all-dp-ranks
#   --restore-file checkpoints/opus-ja2en/i3d/i3d-mask0/checkpoint_best.pt

# --keep-last-epochs $keep_last_epochs
# --keep-best-checkpoints
# --no-epoch-checkpoints
cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task translation 
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --patience $patience
  --keep-last-epochs $keep_last_epochs
  --num-workers 2
  --reset-dataloader
  --log-interval 20"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

if [ -n "$SA_image_dropout" ]; then
cmd=${cmd}" --SA-image-dropout "${SA_image_dropout}
fi
if [ -n "$SA_attention_dropout" ]; then
cmd=${cmd}" --SA-attention-dropout "${SA_attention_dropout}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
