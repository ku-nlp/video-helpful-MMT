import sys
from vizseq.scorers.meteor import METEORScorer
import json

def read_file(path):
    i = 0
    toks = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            toks.append(line)
            i += 1
    return toks, i

hypo_path=sys.argv[1]
sys_toks, i1 = read_file(hypo_path)
ref_toks, i2 = read_file(sys.argv[2])

assert i1 == i2, f"{i1} not equal to {i2} ,error"

translations, ref = [], []
for k in range(i1):
    translations.append(sys_toks[k])
    ref.append(ref_toks[k])

meteor_score = METEORScorer(sent_level=True, corpus_level=True).score(
        translations, [ref]
    )
print(f"The Meteor Score is {meteor_score.corpus_score}")

# sent_scores_path=hypo_path.replace("hypo.sorted", "meteor_sent_scores.json")
# with open(sent_scores_path, 'w') as f:
#     json.dump(meteor_score.sent_scores, f)
# print(f"Save meteor sent scores to {sent_scores_path}")

