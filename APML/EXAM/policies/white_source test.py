def BSM(seq):
    first, second, third = seq[0], seq[1], seq[2]
    max = max(seq[:3])
    max_idx = argmax(seq)
    min = min(seq[:3])

BSM([1,2,3,4])