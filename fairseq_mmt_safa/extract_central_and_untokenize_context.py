from nltk.tokenize.treebank import TreebankWordDetokenizer
from mosestokenizer import *
import sys

# Existing:
# tokenized: three sentences                           sys.argv[1]
# translation result: tokenized three sentences     sys.argv[2]

# Target: (all for test)
# tokenized central sentence                           For meteor score              sys.argv[3]
# tokenized central translation result                 For meteor score              sys.argv[4]
# untokenized central translation result              For bleu score                  sys.argv[5]

# use untokenized 

if __name__=="__main__":
    # input
    f_tok_three=open(sys.argv[1], 'r')
    f_trans_tok_three=open(sys.argv[2], 'r')

    # output
    f_tok_cen=open(sys.argv[3], 'w')
    f_trans_tok_cen=open(sys.argv[4], 'w')
    f_trans_untok_cen=open(sys.argv[5], 'w')
    
    with MosesDetokenizer('en') as detokenize:
        for line in f_trans_tok_three.readlines():
            sents=line.split("<\s>")
            if len(sents)!=3:
                print(line)
            sent=sents[1].strip()
            f_trans_tok_cen.write(sent+'\n')

            sent=sent.replace("&apos;", "'")
            words=sent.split(" ")
#             sent=TreebankWordDetokenizer().detokenize(words)
            sent=detokenize(words)
            f_trans_untok_cen.write(sent+'\n')

        for line in f_tok_three.readlines():
            sents=line.split("<\s>")
            assert len(sents)==3
            sent=sents[1].strip()
            f_tok_cen.write(sent+"\n")
