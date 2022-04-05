import re
import numpy as np
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt

domain_abbrs = ["ATP", "BR", "GW", "PM", "RB", "SCW", "SSE"]

snips_domains = [
    "AddToPlaylist", "BookRestaurant", "GetWeather",
    "PlayMusic", "RateBook", "SearchCreativeWork",
    "SearchScreeningEvent"
]

dyn_mats = {
    "ATP": ["PM", "BR", "GW", "RB", "SCW", "SSE"],
    "BR": ["GW", "SSE", "PM", "ATP", "RB", "SCW"],
    "GW": ["BR", "SSE", "ATP", "PM", "RB", "SCW"],
    "PM": ["ATP", "BR", "GW", "RB", "SCW", "SSE"],
    "RB": ["SCW", "SSE", "ATP", "BR", "GW", "PM"],
    "SCW": ["RB", "SSE", "ATP", "BR", "GW", "PM"],
    "SSE": ["BR", "GW", "RB", "SCW", "ATP", "PM"],
}

lens = {
    "ATP": 1,
    "BR": 3,
    "GW": 2,
    "PM": 2,
    "RB": 2,
    "SCW": 2,
    "SSE": 4,
}


def load_f1_infos(fpath):
    values = []

    with open(fpath, "r") as fr:
        data = fr.read()

    values = re.findall(r"\[FTeF1:(.+)\]", data)
    seen_f1s = [float(v.split(",")[0]) for v in values]
    unseen_f1s = [float(v.split(",")[1]) for v in values]
    f1s = [float(v.split(",")[2]) for v in values]

    seen_f1s = np.array(seen_f1s)
    unseen_f1s = np.array(unseen_f1s)
    f1s = np.array(f1s)
    return seen_f1s, unseen_f1s, f1s


def load_target_infos(fpath):
    values = []

    with open(fpath, "r") as fr:
        data = fr.read()

    values = re.findall(r"target=.+?,", data)
    return values


fpath = "./logs/slotfilling-coach-dyn.log"
values = load_target_infos(fpath)
print(Counter(values))


nd = len(snips_domains)

lrs = [
    0.0001, 0.0001,
    0.0005, 0.0005,
    0.001, 0.001,
    0.005, 0.005,
    0.01, 0.01
]

for name in ["dyn", "coach-dyn"]:
    fpath = "./logs/slotfilling-{}.log".format(name)
    seen_f1s, unseen_f1s, f1s = load_f1_infos(fpath)

    print(f1s.shape)
    f1s = f1s.reshape(nd * (nd - 1), 10)
    seen_f1s = seen_f1s.reshape(nd * (nd - 1), 10)
    unseen_f1s = unseen_f1s.reshape(nd * (nd - 1), 10)

    max_js = np.argmax(f1s, axis=1)

    best_lrs = [lrs[j] for j in max_js]
    print(Counter(best_lrs))

    mean_f1s = []
    mean_seen_f1s = []
    mean_unseen_f1s = []
    for k in range(len(f1s)):
        inds = np.argsort(-1.0 * f1s[k])[0:5]
        mean_f1s.append(f1s[k][inds].mean())
        mean_seen_f1s.append(seen_f1s[k][inds].mean())
        mean_unseen_f1s.append(unseen_f1s[k][inds].mean())

    f1s = np.array(mean_f1s).reshape(nd, nd - 1)
    seen_f1s = np.array(mean_seen_f1s).reshape(nd, nd - 1)
    unseen_f1s = np.array(mean_unseen_f1s).reshape(nd, nd - 1)
    print(f1s.mean(axis=0))

    fig, axes = plt.subplots(nd, 1, figsize=(12, 12))

    for d, dom in enumerate(domain_abbrs):
        ax = axes[d]
        n = lens[dom]

        sizes = [50] * n + [20] * (nd - 1 - n)
        colors = ["Red"] * n + ["Black"] * (nd - 1 - n)

        ys = f1s[d]
        seen_ys = seen_f1s[d]
        unseen_ys = unseen_f1s[d]

        ax.plot(range(nd - 1), ys, linestyle="solid")
        # ax.errorbar(range(nd - 1), mean_ys, yerr=std_ys)
        ax.scatter(range(nd - 1), ys, s=sizes, c=colors)

        ax.set_xticks(range(nd - 1))
        ax.set_xticklabels(dyn_mats[dom])

    plt.savefig("./logs/{}.jpg".format(name), bbox_inches="tight")
    plt.show()

    res = np.concatenate(
        [f1s, seen_f1s, unseen_f1s], axis=0
    )

    df = pd.DataFrame(res)
    df.to_csv("./logs/{}.csv".format(name), header=False, index=False)
