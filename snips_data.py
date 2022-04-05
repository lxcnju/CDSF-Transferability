import os
import copy
import numpy as np

import torch
from torch.utils import data


PAD_INDEX = 0
BIO_PAD_INDEX = 0
BIOSLOT_PAD_INDEX = 0

snips_domains = [
    "AddToPlaylist", "BookRestaurant", "GetWeather",
    "PlayMusic", "RateBook", "SearchCreativeWork",
    "SearchScreeningEvent"
]

domain2int = {domain: i for i, domain in enumerate(snips_domains)}

bio2int = {
    "O": 0,
    "B": 1,
    "I": 2,
}

slot2desp = {
    'playlist': 'playlist',
    'music_item': 'music item',
    'geographic_poi': 'geographic position',
    'facility': 'facility',
    'movie_name': 'moive name',
    'location_name': 'location name',
    'restaurant_name': 'restaurant name',
    'track': 'track',
    'restaurant_type': 'restaurant type',
    'object_part_of_series_type': 'series',
    'country': 'country',
    'service': 'service',
    'poi': 'position',
    'party_size_description': 'person',
    'served_dish': 'served dish',
    'genre': 'genre',
    'current_location': 'current location',
    'object_select': 'this current',
    'album': 'album',
    'object_name': 'object name',
    'state': 'location',
    'sort': 'type',
    'object_location_type': 'location type',
    'movie_type': 'movie type',
    'spatial_relation': 'spatial relation',
    'artist': 'artist',
    'cuisine': 'cuisine',
    'entity_name': 'entity name',
    'object_type': 'object type',
    'playlist_owner': 'owner',
    'timeRange': 'time range',
    'city': 'city',
    'rating_value': 'rating value',
    'best_rating': 'best rating',
    'rating_unit': 'rating unit',
    'year': 'year',
    'party_size_number': 'number',
    'condition_description': 'weather',
    'condition_temperature': 'temperature'
}


def get_slot_embeds(slot_labels, wiki_wv, ngram_wv, args):
    label_embeds = []
    for label in slot_labels:
        label = slot2desp[label]

        ws = label.split()

        embeds1 = wiki_wv.get_vecs_for_tokens(ws)
        embeds2 = ngram_wv.get_vecs_for_tokens(ws)
        embeddings = np.concatenate([embeds1, embeds2], axis=1)

        vec = embeddings.sum(axis=0)
        label_embeds.append(vec)

    label_embeds = np.array(label_embeds)
    return label_embeds


def get_bioslot_embeds(bioslot_labels, wiki_wv, ngram_wv, args):
    replace_dict = {"O": "outside", "B": "begin", "I": "inside"}

    label_embeds = []
    for label in bioslot_labels:
        if label == "O":
            label = replace_dict[label]
        else:
            desp = slot2desp[label.split("-")[1]]
            label = replace_dict[label[0]] + " " + desp

        ws = label.split()

        embeds1 = wiki_wv.get_vecs_for_tokens(ws)
        embeds2 = ngram_wv.get_vecs_for_tokens(ws)
        embeddings = np.concatenate([embeds1, embeds2], axis=1)

        vec = embeddings.sum(axis=0)
        label_embeds.append(vec)

    label_embeds = np.array(label_embeds)
    return label_embeds


def extract_slot_ranges(tags):
    bi = -1
    ei = -1

    ranges = []
    ss = []
    for i, t in enumerate(tags):
        if t[0] == "B":
            s = t.split("-")[1]
            ss.append(s)
            if ei >= 0:
                ranges.append([bi, ei])
            bi = i
            ei = i
        elif t[0] == "I":
            ei = i
        else:
            if ei >= 0:
                ranges.append([bi, ei])
                bi = -1
                ei = -1

    if ei > 0:
        ranges.append([bi, ei])

    assert len(ranges) == len(ss)
    return ss, ranges


def load_snips_txt(fpath, domain):
    tokens = []              # [[xxx, ...], ...], words split by space
    bioslot_tags = []        # [[B-xxx_xxx, I-xxx_xxx, O, ...], ...] for normal
    bio_tags = []            # [[B, I, O, ...], ...] for coach
    slots = []               # [[xxx_xxx, ...], ...] for slots
    slot_ranges = []         # [[[i, j], ...]]
    lens = []

    with open(fpath, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()

            if len(line) <= 0:
                continue

            line = line.split("\t")
            assert len(line) == 2
            tks = line[0].strip().split()
            ts = line[1].strip().split()

            assert len(tks) == len(ts)

            bio_ts = [t[0] for t in ts]

            # find slot
            ss, rs = extract_slot_ranges(ts)

            tokens.append(tks)
            bioslot_tags.append(ts)
            bio_tags.append(bio_ts)
            slots.append(ss)
            slot_ranges.append(rs)
            lens.append(len(tks))

    domains = [domain] * len(tokens)

    data = {
        "tokens": tokens,
        "bioslot_tags": bioslot_tags,
        "bio_tags": bio_tags,
        "slots": slots,
        "slot_ranges": slot_ranges,
        "domains": domains,
        "lens": lens,
    }
    return data


def load_snips_txt_single_domain(fdir, domain):
    tr_fpath = os.path.join(
        fdir, domain, "{}.txt".format(domain)
    )
    seen_fpath = os.path.join(
        fdir, domain, "seen_slots.txt"
    )
    unseen_fpath = os.path.join(
        fdir, domain, "unseen_slots.txt"
    )

    tr_data = load_snips_txt(tr_fpath, domain)
    seen_te_data = load_snips_txt(seen_fpath, domain)
    unseen_te_data = load_snips_txt(unseen_fpath, domain)

    val_data = {}
    for key in tr_data.keys():
        val_data[key] = copy.deepcopy(tr_data[key][0:500])

    te_data = {}
    for key in seen_te_data.keys():
        te_data[key] = copy.deepcopy(seen_te_data[key])
        te_data[key].extend(unseen_te_data[key])

    data = {
        "tr_data": tr_data,
        "val_data": val_data,
        "te_data": te_data,
        "seen_te_data": seen_te_data,
        "unseen_te_data": unseen_te_data,
    }
    return data


def trans_data(vdict, vocab2int, slot2int, bioslot2int):
    uni_slot_ids = []
    uni_bioslot_ids = []

    tk_ids = []
    for es in vdict["tokens"]:
        tk_ids.append([vocab2int[e] for e in es])

    bioslot_tag_ids = []
    for es in vdict["bioslot_tags"]:
        ids = [bioslot2int[e] for e in es]
        bioslot_tag_ids.append(ids)
        uni_bioslot_ids.extend(ids)

    bio_tag_ids = []
    for es in vdict["bio_tags"]:
        bio_tag_ids.append([bio2int[e] for e in es])

    slot_ids = []
    for es in vdict["slots"]:
        ids = [slot2int[e] for e in es]
        slot_ids.append(ids)
        uni_slot_ids.extend(ids)

    dom_ids = [domain2int[d] for d in vdict["domains"]]

    uni_slot_ids = list(sorted(np.unique(uni_slot_ids)))
    uni_bioslot_ids = list(sorted(np.unique(uni_bioslot_ids)))

    data = {
        "tk_ids": tk_ids,
        "bioslot_tag_ids": bioslot_tag_ids,
        "bio_tag_ids": bio_tag_ids,
        "slot_ids": slot_ids,
        "slot_ranges": vdict["slot_ranges"],
        "dom_ids": dom_ids,
        "lens": vdict["lens"],
        "uni_slot_ids": uni_slot_ids,
        "uni_bioslot_ids": uni_bioslot_ids,
    }

    return data


def load_snips_data(fdir):
    all_words = []
    all_slots = []          # music_item, ...
    all_bioslots = []       # B-music_item, ...

    domain2data = {}
    for domain in snips_domains:
        data = load_snips_txt_single_domain(fdir, domain)
        domain2data[domain] = data

        # add to vocab
        for tks in data["tr_data"]["tokens"]:
            all_words.extend(tks)

        for bioslots in data["tr_data"]["bioslot_tags"]:
            all_bioslots.extend(bioslots)

        for slots in data["tr_data"]["slots"]:
            all_slots.extend(slots)

            for slot in slots:
                all_words.extend(slot.split("_"))

    # vocab, slot2int, bioslot2int
    vocab = ["[PAD]", "[UNK]"] + list(np.unique(all_words))
    print("Length of vocab: {}".format(len(vocab)))

    slots = list(sorted(np.unique(all_slots)))
    print("Total number of slots: {}".format(len(slots)))
    print("Slots example: ", slots[0:5])

    bioslots = [t for t in list(sorted(np.unique(all_bioslots))) if t != "O"]
    bioslots = ["O"] + bioslots
    print("Total number of slots: {}".format(len(bioslots)))
    print("BIO-Slots example: ", bioslots[0:5])

    vocab2int = {w: i for i, w in enumerate(vocab)}
    slot2int = {s: i for i, s in enumerate(slots)}
    bioslot2int = {s: i for i, s in enumerate(bioslots)}
    print("PadIndex: ", vocab2int["[PAD]"])
    print("BIOPadIndex: ", bio2int["O"])
    print("BIOSlotPadIndex: ", bioslot2int["O"])

    # transform data in each domain
    all_dom_data = {}
    for domain in snips_domains:
        dom_data = {}
        for key, vdict in domain2data[domain].items():
            dom_data[key] = trans_data(
                vdict, vocab2int, slot2int, bioslot2int
            )

        # dom2slot mask, dom2bioslot_mask
        n_slots = len(slots)
        slot_mask = np.zeros(n_slots)
        inds = dom_data["tr_data"]["uni_slot_ids"]
        slot_mask[inds] = 1.0

        n_bioslots = len(bioslots)
        bioslot_mask = np.zeros(n_bioslots)
        inds = dom_data["tr_data"]["uni_bioslot_ids"]
        bioslot_mask[inds] = 1.0

        dom_data["slot_mask"] = slot_mask
        dom_data["bioslot_mask"] = bioslot_mask

        all_dom_data[domain] = dom_data

        print("Domain:{}, NofSlot:{}, NofBIOSlot:{}".format(
            domain, slot_mask.sum(), bioslot_mask.sum()
        ))

    all_dom_data["vocab2int"] = vocab2int
    all_dom_data["slot2int"] = slot2int
    all_dom_data["bioslot2int"] = bioslot2int
    all_dom_data["uni_slots"] = slots
    all_dom_data["uni_bioslots"] = bioslots

    return all_dom_data


class SnipsDataset(data.Dataset):
    def __init__(
        self,
        indices,
        bioslot_tags,
        bio_tags,
        slots,
        slot_ranges,
        domains,
    ):
        self.indices = indices
        self.bioslot_tags = bioslot_tags
        self.bio_tags = bio_tags
        self.slots = slots
        self.slot_ranges = slot_ranges
        self.domains = domains
        self.slot_nums = [len(rs) for rs in slot_ranges]
        self.lens = [len(inds) for inds in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        inds = self.indices[k]
        bioslot_ts = self.bioslot_tags[k]
        bio_ts = self.bio_tags[k]
        slot = self.slots[k]
        slot_rs = self.slot_ranges[k]
        slot_num = self.slot_nums[k]
        dom = self.domains[k]
        length = self.lens[k]

        return inds, bioslot_ts, bio_ts, slot, slot_rs, slot_num, dom, length


def collate_fn(data):
    inds, bioslot_tag_ids, bio_tag_ids, slots, slot_ranges, \
        slot_nums, doms, lens = zip(*data)

    bs = len(inds)
    max_len = max(lens)

    padded_inds = torch.LongTensor(
        bs, max_len
    ).fill_(PAD_INDEX)

    padded_bioslot_tag_ids = torch.LongTensor(
        bs, max_len
    ).fill_(BIOSLOT_PAD_INDEX)

    padded_bio_tag_ids = torch.LongTensor(
        bs, max_len
    ).fill_(BIO_PAD_INDEX)

    for b in range(bs):
        L = lens[b]
        padded_inds[b, :L] = torch.LongTensor(inds[b])
        padded_bioslot_tag_ids[b, :L] = torch.LongTensor(bioslot_tag_ids[b])
        padded_bio_tag_ids[b, :L] = torch.LongTensor(bio_tag_ids[b])

    max_n_slot = max(slot_nums)
    padded_slots = []
    for ss in slots:
        padded_slots.append(ss + [-1] * (max_n_slot - len(ss)))

    padded_slot_ranges = []
    for rs in slot_ranges:
        padded_slot_ranges.append(rs + [[-1, -1]] * (max_n_slot - len(rs)))

    padded_slots = torch.LongTensor(padded_slots)
    padded_slot_ranges = torch.LongTensor(padded_slot_ranges)
    slot_nums = torch.LongTensor(slot_nums)
    doms = torch.LongTensor(doms)
    lens = torch.LongTensor(lens)

    return padded_inds, padded_bioslot_tag_ids, padded_bio_tag_ids, \
        padded_slots, padded_slot_ranges, slot_nums, doms, lens
