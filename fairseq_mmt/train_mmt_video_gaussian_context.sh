#! /usr/bin/bash
set -e

# device=0,1,2,3,4,5,6,7
device=$1
# image_feat=i3d1
patience=20
max_tokens=4000 # for c4c
fp16=0 #0
lambda=0.5
inverse_T=1
weight=0.5
order=$2
image_feat=c4c_w${weight}_l${lambda}_it${inverse_T}_weighted_gaussian_dist2_context_${order}

task=opus-ja2en
mask_data=mask0
tag=$image_feat/$image_feat-$mask_data
save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'opus-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opus.ja-en-context
fi

decrease=0
criterion=label_smoothed_cross_entropy_with_gaussian_context
amp=0
lr=0.005
warmup=2000
update_freq=1
keep_last_epochs=10
max_update=800000
dropout=0.3

arch=image_multimodal_transformer_SA_top
SA_attention_dropout=0.1
SA_image_dropout=0.1

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
elif [[ $image_feat == *"i3d"* ]] ; then
# 	image_feat_path=/dataset/OpusEJ/OpusEJ_i3d_feature
	if [[ $HOSTNAME == agni  ]]; then
		image_feat_path=/data/OpusEJ_i3d_feature
		echo "image_feat_path=$image_feat_path"
	else
		image_feat_path=/home/dataset/OpusEJ_i3d_feature
	fi
	image_feat_dim=2048 # (32, 2048)
	image_feat_whole_dim="32 2048" # (32, 2048)
elif [[ $image_feat == *"c4c"* ]]; then
	image_feat_path=/data/OpusEJ_c4c_feature
# 	image_feat_path=/dataset/OpusEJ/OpusEJ_c4c_feature
	image_feat_dim=512
	image_feat_whole_dim="12 512" # (12, 512)
elif [[ $image_feat == *"videoMAE"* ]]; then
	if [[ $HOSTNAME == agni  ||  $HOSTNAME == kubera  ||  $HOSTNAME == saffron4 ||  $HOSTNAME == saffron ||  $HOSTNAME == moss110 ]]; then
		image_feat_path=/data/OpusEJ_videoMAE_feature_224
		echo "image_feat_path=$image_feat_path"
	fi
	image_feat_dim=384
	image_feat_whole_dim="1568 384" # (1568, 384)
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
#   --data-buffer-size 1
#   --restore-file checkpoints/opus-ja2en/i3d/i3d-mask0/checkpoint_best.pt
#   --keep-best-epochs $keep_last_epochs
cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1 --balancing-lambda $lambda --inverse-softmax-tempreature $inverse_T --decrease-addictive-object $decrease --weight $weight
  --task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --image-feat-whole-dim $image_feat_whole_dim
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq --max-update $max_update
  --patience $patience
  --keep-last-epochs $keep_last_epochs
  --num-workers 2
  --log-interval 20"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

if [ $amp -eq 1 ]; then
cmd=${cmd}" --amp "
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
