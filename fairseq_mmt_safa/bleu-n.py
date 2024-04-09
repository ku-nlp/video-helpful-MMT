import sys
import math
# BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.5.1 = 15.38 47.6/20.0/12.9/8.8 (BP = 0.848 ratio = 0.858 hyp_len = 7887 ref_len = 9190)
for line in sys.stdin:
    sys.stdout.write(line)
#     BP=float(line.split(" ")[6])
    reflen=int(line.split(" ")[-1][:-2])
    hyplen=int(line.split(" ")[-4])
    BP=math.exp(1-reflen/hyplen)
    print(BP)
    grams=[float(i) for i in line.split(" ")[3].split("/")]
    for i in range(4):
        total=1
        for j in range(i+1):
            total*=grams[j]
        total=total**(1/(i+1))*BP
        print(f"BLEU-{i+1}: {total:.2f}")
    