#!/usr/bin/bash
set -e

_mask=mask0
_image_feat=$1

# set device
gpu=0

model_root_dir=checkpoints

# set task
task=opus-zh2en
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
elif [ $task == 'opus-zh2en' ]; then
	src_lang=zh
	tgt_lang=en
	data_dir=opus.zh-en-context
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
elif [ $image_feat == "i3d" ] || [ $image_feat == "i3d_random" ]|| [ $image_feat == "i3d_random1" ]; then
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
fi

# data set
ensemble=5
batch_size=100
beam=5

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble --use-best 0
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
  --task translation
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
elif [ $task == "opus-zh2en" ] && [ $who == 'test' ]; then
	ref=data/opus-zh-en-context/test.en
fi	

hypo=$model_dir/hypo.sorted

python3 extract_central_and_untokenize_context.py $ref $hypo data/opus-zh-en-context/central_test.en $model_dir/central_hypo.sorted $model_dir/central_hypo.sorted.untokenized

python3 meteor.py $model_dir/central_hypo.sorted data/opus-zh-en-context/central_test.en > $model_dir/meteor_$who.log
cat $model_dir/meteor_$who.log

sacrebleu data/opus-zh-en-context.untokenized/test.en -i $model_dir/central_hypo.sorted.untokenized -w 2 | python bleu-n.py
