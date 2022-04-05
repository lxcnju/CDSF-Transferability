import os
import scipy.stats
import numpy as np
import pandas as pd

from paths import snips_fdir

import matplotlib
from matplotlib import pyplot as plt

from snips_data import load_snips_data


snips_domains = [
    "AddToPlaylist", "BookRestaurant", "GetWeather",
    "PlayMusic", "RateBook", "SearchCreativeWork",
    "SearchScreeningEvent"
]

domain_abbrs = ["ATP", "BR", "GW", "PM", "RB", "SCW", "SSE"]


slot_abbrs_dict = {
    "album": "Album",
    "artist": "Artist",
    "best rating": "Best Rat.",
    "city": "City",
    "condition description": "Con. Des.",
    "condition temperature": "Con. Tem.",
    "country": "Country",
    "cuisine": "Cuisine",
    "current location": "Cur. Loc.",
    "entity name": "Ent. Name",
    "facility": "Facility",
    "genre": "Genre",
    "geographic poi": "Geo. Poi.",
    "location name": "Loc. Name",
    "movie name": "Mov. Name",
    "movie type": "Mov. Type",
    "music item": "Music It.",
    "object location type": "Obj. Loc. T.",
    "object name": "Obj. Name",
    "object part of series type": "Obj. POS.",
    "object select": "Obj. Sel.",
    "object type": "Obj. Type",
    "party size description": "Par. S. D.",
    "party size number": "Par. S. N.",
    "playlist": "Playlist",
    "playlist owner": "Playlist O.",
    "poi": "Poi",
    "rating unit": "Rat. Unit",
    "rating value": "Rat. Val.",
    "restaurant name": "Res. Name",
    "restaurant type": "Res. Type",
    "served dish": "Served Dish",
    "service": "Service",
    "sort": "Sort",
    "spatial relation": "Spa. Rel.",
    "state": "State",
    "timeRange": "TimeRange",
    "track": "Track",
    "year": "Year",
}


data = load_snips_data(snips_fdir)

domain2slot = []
for domain in snips_domains:
    domain2slot.append(data[domain]["slot_mask"])
domain2slot = np.array(domain2slot)


uni_slots = data["uni_slots"]
slot2int = data["slot2int"]
slot_abbrs = [
    slot_abbrs_dict[" ".join(slot.split("_"))] for slot in uni_slots
]

print(len(uni_slots))
print(len(np.unique(slot_abbrs)))

for slot, abbr in zip(uni_slots, slot_abbrs):
    print(r'"{}": "{}",'.format(slot, abbr))

nd = len(snips_domains)
ns = len(uni_slots)

cnts = domain2slot

fig, ax = plt.subplots(figsize=(8, 4))

cmap = plt.get_cmap(name="YlGn", lut=2)
im = ax.imshow(cnts, cmap=cmap)

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.25, ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(["0", "1"])

ax.set_xticks(np.arange(ns))
ax.set_yticks(np.arange(nd))
ax.set_xticklabels(slot_abbrs)
ax.set_yticklabels(domain_abbrs)

ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

ax.set_xticks(np.arange(ns + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(nd + 1) - 0.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)


fig.tight_layout()
plt.savefig("./logs/domain_slot.jpg", bbox_inches="tight")
plt.show()

df = pd.DataFrame(cnts)
df.to_csv("./logs/domain_slot.csv", header=False, index=False)


# cross domain
co_cnts = np.dot(cnts, cnts.transpose())

fig, ax = plt.subplots(figsize=(4, 4))

cmap = plt.get_cmap(name="YlGn", lut=15)
im = ax.imshow(co_cnts, cmap=cmap)

yticks = [1.0 / 15.0 * i + 1.0 / 30.0 for i in range(15)]

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, ticks=yticks)

yticklabels = [str(i) for i in range(15)]
cbar.ax.set_yticklabels(yticklabels)

ax.set_xticks(np.arange(nd))
ax.set_yticks(np.arange(nd))
ax.set_xticklabels(domain_abbrs)
ax.set_yticklabels(domain_abbrs)

ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

ax.set_xticks(np.arange(nd + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(nd + 1) - 0.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)

for i in range(nd):
    for j in range(nd):
        text = im.axes.text(
            j, i,
            "{}".format(int(co_cnts[i, j])),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
        )


df = pd.DataFrame(co_cnts)
df.to_csv("./logs/coslot.csv", index=False, header=False)

fig.tight_layout()
plt.savefig("./logs/coslot.jpg", bbox_inches="tight")
plt.show()


# dist
dists = []
for domain in snips_domains:
    slots = []
    for ss in data[domain]["tr_data"]["slot_ids"]:
        slots.extend(ss)

    slots = np.array(slots)
    dist = [np.mean(slots == s) for s in range(len(uni_slots))]
    dists.append(dist)
dists = np.array(dists)

# L1 distance
res = np.zeros((nd, nd))
for i, source in enumerate(snips_domains):
    for j, target in enumerate(snips_domains):
        res[i, j] = np.sum(np.abs(dists[i] - dists[j]))

fig, ax = plt.subplots(figsize=(4, 4))

cmap = plt.get_cmap(name="YlGn")
im = ax.imshow(res, cmap=cmap)

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(np.arange(nd))
ax.set_yticks(np.arange(nd))
ax.set_xticklabels(domain_abbrs)
ax.set_yticklabels(domain_abbrs)

ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

ax.set_xticks(np.arange(nd + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(nd + 1) - 0.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)

for i in range(nd):
    for j in range(nd):
        text = im.axes.text(
            j, i,
            "{:.2f}".format(res[i, j]),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
        )

df = pd.DataFrame(res)
df.to_csv("./logs/l1discrepancy.csv", index=False, header=False)

fig.tight_layout()
plt.savefig("./logs/l1discrepancy.jpg", bbox_inches="tight")
plt.show()


# number
nums = []
for domain in snips_domains:
    slots = []
    for ss in data[domain]["tr_data"]["slot_ids"]:
        slots.extend(ss)

    slots = np.array(slots)
    nus = [np.sum(slots == s) for s in range(len(uni_slots))]
    nums.append(nus)
nums = np.array(nums)

fig, ax = plt.subplots(figsize=(8, 4))

cmap = plt.get_cmap(name="YlGn")
im = ax.imshow(nums, cmap=cmap)

cbar = ax.figure.colorbar(im, ax=ax, shrink=0.25)

ax.set_xticks(np.arange(ns))
ax.set_yticks(np.arange(nd))
ax.set_xticklabels(slot_abbrs)
ax.set_yticklabels(domain_abbrs)

ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)

plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

ax.set_xticks(np.arange(ns + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(nd + 1) - 0.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)


fig.tight_layout()
plt.savefig("./logs/domain_slot_nums.jpg", bbox_inches="tight")
plt.show()

df = pd.DataFrame(nums)
df.to_csv("./logs/domain_slot_nums.csv", header=False, index=False)
