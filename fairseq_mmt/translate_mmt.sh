#!/usr/bin/bash
set -e

_mask=mask0
_image_feat=$1

# set device
gpu=0

model_root_dir=checkpoints

# set task
task=opus-random-ja2en
mask_data=$_mask
image_feat=$_image_feat

who=test	#test1, test2
random_image_translation=0 #1
length_penalty=0.8
use_best=0
ensemble_type=last

# set tag
model_dir_tag=$image_feat/$image_feat-$mask_data

if [ $task == 'opus-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opus.ja-en
elif [ $task == 'opus-random-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=opus-random.ja-en
elif [ $task == 'transet-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=transet.ja-en
elif [ $task == 'viop-ja2en' ]; then
	src_lang=ja
	tgt_lang=en
	data_dir=viop.ja-en
elif [ $task == 'opus-zh2en' ]; then
	src_lang=zh
	tgt_lang=en
	data_dir=opus.zh-en
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
	if [[ $src_lang == "ja" ]]; then # ==前后要有空格
		image_feat_path=/data/OpusEJ_c4c_feature
	elif [[ $src_lang == "zh" ]]; then
		image_feat_path=/data/OpusZE_c4c_feature
	fi
	echo "image_feat_path=$image_feat_path"
	image_feat_dim=512
	image_feat_whole_dim="12 512" # (12, 512)
elif [[ $image_feat == *"videoMAE"* ]]; then
	image_feat_path=/data/OpusEJ_videoMAE_feature_224
	image_feat_dim=384
	image_feat_whole_dim="1568 384" # (1568, 384)
elif [[ $image_feat == *"detr"* ]]; then
	if [[ $src_lang == "ja" ]]; then # ==前后要有空格
		image_feat_path=/data/OpusEJ_central_frame_detr_feature
	elif [[ $src_lang == "zh" ]]; then
		image_feat_path=/data/OpusZE_central_frame_detr_feature
	fi
	echo "image_feat_path=$image_feat_path"
	image_feat_dim=256
	image_feat_whole_dim="100 256" # (1568, 384)
fi

# data set
ensemble=5
batch_size=128
beam=5

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/$ensemble_type$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/$ensemble_type$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble --use-best $use_best
        fi
        checkpoint=$ensemble_type$ensemble.ensemble.pt
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

if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
	ref=data/multi30k/test.2016.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
	ref=data/multi30k/test.2017.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
	ref=data/multi30k/test.coco.de

elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
	ref=data/multi30k/test.2016.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
	ref=data/multi30k/test.2017.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
	ref=data/multi30k/test.coco.fr
elif [ $task == "opus-ja2en" ] && [ $who == 'test' ]; then
	ref=data/opus-ja-en/test.en
elif [ $task == "opus-random-ja2en" ] && [ $who == 'test' ]; then
	ref=data/opus-random-ja-en/test.en
elif [ $task == "transet-ja2en" ] && [ $who == 'test' ]; then
	ref=data/transet-ja-en/test.en
elif [ $task == "viop-ja2en" ] && [ $who == 'test' ]; then
	ref=data/viop-ja-en/test.en
elif [ $task == "opus-zh2en" ] && [ $who == 'test' ]; then
	ref=data/opus-zh-en/test.en
fi	

hypo=$model_dir/hypo.sorted
python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log
# echo  "python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log"
cat $model_dir/meteor_$who.log

python3 untokenize_en.py $hypo $model_dir/hypo.sorted.untokenized
sacrebleu data/opus-random-ja-en.untokenized/test.en -i $model_dir/hypo.sorted.untokenized -w 2 | python bleu-n.py
