import numpy as np


def get_pseudo_count(pg):
    eps = 1e-6

    exp = np.exp(max(0, pg))
    cnt = (exp - 1 + eps) ** (- 1)

    return cnt


def get_counters(schedule, pgs, num_classes):
    real_counters = [[0] for _ in range(num_classes)]
    pseudo_counters = [[0] for _ in range(num_classes)]
    pgs_by_classes = [[] for _ in range(num_classes)]

    for i in range(len(schedule)):
        cl = schedule[i]
        pgs_by_classes[cl].append(pgs[i])

        for j in range(num_classes):
            if j == cl:
                real_counters[j].append(real_counters[j][-1] + 1)
                pseudo_counters[j].append(get_pseudo_count(pgs[i]))
            else:
                real_counters[j].append(real_counters[j][-1])
                pseudo_counters[j].append(pseudo_counters[j][-1])

    real_counters = np.array(real_counters)
    pseudo_counters = np.array(pseudo_counters)
    pgs_by_classes = np.array(pgs_by_classes)

    return real_counters, pseudo_counters, pgs_by_classes