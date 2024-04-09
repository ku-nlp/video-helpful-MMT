src='zh'
tgt='en'

TEXT=data/opus-$src-$tgt-context

# In text, each line is a sentence
rm -rf /home/code/fairseq_mmt_safa/data-bin/opus.zh-en-context

fairseq-preprocess --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/opus.$src-$tgt-context \
  --thresholdtgt 3 \
  --thresholdsrc 3 \
  --workers 8 
  
  # For EVA, threshold is 3
  # For VISA, threshold is 1
  # For transet, threshold is 2

