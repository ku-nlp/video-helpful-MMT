#!/usr/bin/bash
set -e

_mask=mask0
_image_feat=$1

# set device
gpu=2

model_root_dir=checkpoints

# set task
task=opus-ja2en
mask_data=$_mask
image_feat=$_image_feat

who=test	#test1, test2
random_image_translation=0 #1
length_penalty=0.8

# set tag
model_dir_tag=$image_feat/$image_feat-$mask_data

if [ $task == 'opus-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opus.ja-en-context
fi


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
	image_feat_path=/data/OpusEJ_i3d_feature
	image_feat_dim=2048 # (32, 2048)
elif [[ $image_feat == *"c4c"* ]]; then
	image_feat_path=/data/OpusEJ_c4c_feature
	image_feat_dim=512
	image_feat_whole_dim="12 512" # (12, 512)
elif [[ $image_feat == *"videoMAE"* ]]; then
	image_feat_path=/data/OpusEJ_videoMAE_feature_224
	image_feat_dim=384
	image_feat_whole_dim="1568 384" # (1568, 384)
fi

# data set
ensemble=5
batch_size=400
beam=5
src_lang=ja

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation_$who.log

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint 
  --gen-subset $who 
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --quiet --remove-bpe
  --task image_mmt
  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --output $model_dir/hypo.txt" 

echo ${cmd}
 
if [ $random_image_translation -eq 1 ]; then
cmd=${cmd}" --random-image-translation "
fi

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $task == "opus-ja2en" ] && [ $who == 'test' ]; then
	ref=data/opus-ja-en-context/test.en
fi	

hypo=$model_dir/hypo.sorted

python3 extract_central_and_untokenize_context.py $ref $hypo data/opus-ja-en-context/central_test.en $model_dir/central_hypo.sorted $model_dir/central_hypo.sorted.untokenized

python3 meteor.py $model_dir/central_hypo.sorted data/opus-ja-en-context/central_test.en > $model_dir/meteor_$who.log
cat $model_dir/meteor_$who.log

sacrebleu data/opus-ja-en-context.untokenized/test.en -i $model_dir/central_hypo.sorted.untokenized -w 2 | python bleu-n.py
