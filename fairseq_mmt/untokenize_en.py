from nltk.tokenize.treebank import TreebankWordDetokenizer
from mosestokenizer import *
import sys

if __name__=="__main__":
    fref=open(sys.argv[1], 'r')
    with open(sys.argv[2], 'w') as fout:
        with MosesDetokenizer('en') as detokenize:
            for line in fref.readlines():
                line=line.rstrip()
                sent=line.replace("&apos;", "'")
                words=sent.split(" ")
#                 sent=TreebankWordDetokenizer().detokenize(words)
                sent=detokenize(words)
                fout.write(sent+'\n')
