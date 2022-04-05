import os
import copy
import random
import argparse
from collections import namedtuple
import numpy as np

import torch

from model import SlotFillingModel

from slot_filling import SlotFilling

from snips_data import load_snips_data
from snips_data import SnipsDataset
from snips_data import snips_domains
from snips_data import get_bioslot_embeds

from paths import save_dir
from paths import snips_fdir
from paths import wv_dir

from text import PretrainWV

from utils import set_gpu


import warnings
warnings.filterwarnings("ignore")


dyn_mats = {
    "ATP": ["PM", "BR", "GW", "RB", "SCW", "SSE"],
    "BR": ["GW", "SSE", "PM", "ATP", "RB", "SCW"],
    "GW": ["BR", "SSE", "ATP", "PM", "RB", "SCW"],
    "PM": ["ATP", "BR", "GW", "RB", "SCW", "SSE"],
    "RB": ["SCW", "SSE", "ATP", "BR", "GW", "PM"],
    "SCW": ["RB", "SSE", "ATP", "BR", "GW", "PM"],
    "SSE": ["BR", "GW", "RB", "SCW", "ATP", "PM"],
}

domain_abbrs = ["ATP", "BR", "GW", "PM", "RB", "SCW", "SSE"]

abbr2domain = {
    abbr: domain for domain, abbr in zip(snips_domains, domain_abbrs)
}


def main(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    if args.cuda:
        print(args.gpu)
        set_gpu(args.gpu)
        torch.backends.cudnn.benchmark = True

    # source and target domains
    data = load_snips_data(snips_fdir)

    # parse data
    src_tr_data = {}
    for source in args.sources:
        for key, values in data[source]["tr_data"].items():
            if key in src_tr_data:
                src_tr_data[key].extend(values)
            else:
                src_tr_data[key] = copy.deepcopy(values)

    tgt_val_data = data[args.target]["val_data"]
    tgt_seen_te_data = data[args.target]["seen_te_data"]
    tgt_unseen_te_data = data[args.target]["unseen_te_data"]

    train_test_data_list = [
        src_tr_data, tgt_val_data,
        tgt_seen_te_data, tgt_unseen_te_data
    ]

    dsets = []
    for ttd in train_test_data_list:
        dset = SnipsDataset(
            indices=ttd["tk_ids"],
            bioslot_tags=ttd["bioslot_tag_ids"],
            bio_tags=ttd["bio_tag_ids"],
            slots=ttd["slot_ids"],
            slot_ranges=ttd["slot_ranges"],
            domains=ttd["dom_ids"],
        )
        dsets.append(dset)

    # slots
    vocab2int = data["vocab2int"]
    uni_bioslots = data["uni_bioslots"]
    bioslot2int = data["bioslot2int"]

    # domain masks
    domain_bioslot_masks = []
    for domain in snips_domains:
        domain_bioslot_masks.append(data[domain]["bioslot_mask"])
    domain_bioslot_masks = np.array(domain_bioslot_masks)

    # pretrained word embeddings
    wiki_wv = PretrainWV(wv_dir, "wikien_300")
    ngram_wv = PretrainWV(wv_dir, "ngram_100")

    # slot embeddings
    bioslot_embeds = get_bioslot_embeds(
        uni_bioslots, wiki_wv, ngram_wv, args
    )

    # load embedding
    model = SlotFillingModel(
        label_embeds=bioslot_embeds,
        args=args
    )

    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    int2vocab = {i: word for word, i in vocab2int.items()}
    vocab = [int2vocab[i] for i in range(len(int2vocab))]
    print("Size of vocabulary: {}".format(len(vocab)))

    total, cnt = wiki_wv.get_cnts_in_pretrain(vocab)
    print("Total tokens: {}, In pretrain: {}.".format(total, cnt))

    embeds1 = wiki_wv.get_vecs_for_tokens(vocab)
    embeds2 = ngram_wv.get_vecs_for_tokens(vocab)
    embeddings = np.concatenate([embeds1, embeds2], axis=1)

    model.load_w2v(embeddings)
    print("Embedding.shape={}".format(embeddings.shape))

    if args.cuda:
        model.cuda()

    # construct datasets/dataloader
    print(args.sources, args.target)

    task = SlotFilling(
        train_set=dsets[0],
        val_set=dsets[1],
        test_set1=dsets[2],
        test_set2=dsets[3],
        domain_bioslot_masks=domain_bioslot_masks,
        bioslot2int=bioslot2int,
        model=model,
        args=args,
    )

    task.main()

    fpath = os.path.join(
        save_dir, args.fname
    )
    task.save_logs(fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()
    gpu = args.gpu
    print("Setting gpu:", gpu)

    param_dict = {
        "dataset": ["snips"],
        "source": ["AddToPlaylist"],
        "target": ["AddToPlaylist"],
        "use_crf": [False],
        "clf": ["clf"],
        "pool": ["lstm"],
        "n_bioslots": [72],
        "n_vocab": [12500],
        "w_dim": [400],
        "hidden_size": [200],
        "epoches": [50],
        "batch_size": [32],
        "optimizer": ["Adam"],
        "lr": [5e-4],
        "lr_mu": [0.0],
        "scheduler": ["StepLR"],
        "step_size": [100],
        "gamma": [1.0],
        "weight_decay": [1e-5],
        "grad_max_norm": [50.0],
        "cuda": [True],
        "gpu": [gpu],
        "fname": ["slotfilling-dyn.log"],
    }

    for use_crf, clf in [(False, "fixle")]:
        for target_abbr in domain_abbrs:
            source_abbrs = dyn_mats[target_abbr]
            for ind in range(len(source_abbrs)):
                sources = [
                    abbr2domain[a] for a in source_abbrs[0:ind + 1]
                ]
                target = abbr2domain[target_abbr]
                for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
                    accs = []
                    for _ in range(2):
                        para_dict = {}
                        for k, vs in param_dict.items():
                            para_dict[k] = random.choice(vs)

                        para_dict["use_crf"] = use_crf
                        para_dict["clf"] = clf
                        para_dict["sources"] = sources
                        para_dict["target"] = target
                        para_dict["lr"] = lr

                        para_dict["fname"] = "slotfilling-dyn.log"

                        main(para_dict)
