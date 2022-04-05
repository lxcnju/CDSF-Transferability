from tqdm import tqdm
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from seqeval.metrics import f1_score

from utils import Averager
from utils import append_to_logs

from tools import construct_group_optimizer
from tools import construct_lr_scheduler

from snips_data import collate_fn


def construct_loader(dset, is_train, args):
    loader = data.DataLoader(
        dset, batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=is_train, drop_last=False
    )
    return loader


class SlotFillingCoach():
    def __init__(
        self,
        train_set,
        val_set,
        test_set1,
        test_set2,
        model,
        domain_slot_masks,
        args
    ):
        self.train_loader = construct_loader(
            train_set, True, args
        )
        self.val_loader = construct_loader(
            val_set, False, args
        )
        self.test_loader1 = construct_loader(
            test_set1, False, args
        )
        self.test_loader2 = construct_loader(
            test_set2, False, args
        )

        self.domain_slot_masks = torch.FloatTensor(domain_slot_masks)
        self.model = model
        self.args = args

        self.optimizer = construct_group_optimizer(
            self.model, args
        )
        self.lr_scheduler = construct_lr_scheduler(
            self.optimizer, args
        )

        self.logs = []

    def main(self):
        best_model = copy.deepcopy(self.model)
        best_epoch = 0
        best_val_f1 = 0.0
        cnt = 0

        for epoch in range(1, self.args.epoches + 1):
            train_loss = self.train(
                model=self.model,
                optimizer=self.optimizer,
                loader=self.train_loader,
                args=self.args
            )
            val_f1 = self.val(
                model=self.model,
                loader=self.val_loader,
                args=self.args,
            )
            _, _, test_f1 = self.test(
                model=self.model,
                loader1=self.test_loader1,
                loader2=self.test_loader2,
                args=self.args,
            )

            self.lr_scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                cnt = 0
            else:
                cnt += 1

            print("[{},{:.5f}] [VF1:{:.4f},{:.4F}] [TF1:{:.4f}]".format(
                epoch, train_loss, val_f1, best_val_f1, test_f1
            ))

            log_str = "[{},{:.5f}] [VF1:{:.4f},{:.4F}] [TF1:{:.4f}]".format(
                epoch, train_loss, val_f1, best_val_f1, test_f1
            )
            self.logs.append(log_str)

            if cnt > 0:
                print("Do not found better model: {}/5".format(cnt))

            if cnt >= 5:
                break

        test_seen_f1, test_unseen_f1, test_f1 = self.test(
            model=best_model,
            loader1=self.test_loader1,
            loader2=self.test_loader2,
            args=self.args
        )
        print("[BeEp:{}] [BeValF1:{:.4f}] [TeF1:{:.4F},{:.4f},{:.4f}]".format(
            best_epoch, best_val_f1, test_seen_f1, test_unseen_f1, test_f1
        ))

        log_str = "[Ep:{}] [FBVF1:{:.4f}] [FTeF1:{:.4F},{:.4f},{:.4f}]".format(
            best_epoch, best_val_f1, test_seen_f1, test_unseen_f1, test_f1
        )
        self.logs.append(log_str)

    def train(self, model, optimizer, loader, args):
        model.train()

        avg_loss = Averager()

        criterion = nn.CrossEntropyLoss(reduction="none")
        for batch in tqdm(loader):
            if args.cuda:
                batch = [elem.cuda() for elem in batch]

                batch_indices = batch[0]
                # batch_bioslot_tsg_ids = batch[1]
                batch_tag_labels = batch[2]
                batch_slot_labels = batch[3]
                batch_slot_ranges = batch[4].detach().cpu().numpy()
                batch_slot_nums = batch[5].detach().cpu().numpy()
                batch_domains = batch[6].detach().cpu().numpy()
                batch_lens = batch[7].detach().cpu().numpy().reshape(-1)

            batch_slot_labels = [
                batch_slot_labels[i][0:j] for i, j in enumerate(
                    batch_slot_nums
                )
            ]
            batch_slot_ranges = [
                batch_slot_ranges[i][0:j] for i, j in enumerate(
                    batch_slot_nums
                )
            ]

            bs = len(batch_indices)

            # tagging
            batch_hs, batch_tag_logits = model.forward_tagging(
                batch_indices
            )

            if self.args.use_crf:
                tag_loss = model.crf.loss(batch_tag_logits, batch_tag_labels)
            else:
                bs = batch_tag_logits.shape[0]
                n_seq = batch_tag_logits.shape[1]
                n_tag = batch_tag_logits.shape[-1]

                batch_tag_logits = batch_tag_logits.reshape((-1, n_tag))
                batch_tag_labels = batch_tag_labels.reshape(-1)

                losses = criterion(batch_tag_logits, batch_tag_labels)
                losses = losses.reshape(bs, n_seq)

                tag_loss = torch.stack([
                    losses[i][0:L].mean() for i, L in enumerate(batch_lens)
                ], dim=0).mean()

            # slot filling
            batch_slot_logits = model.forward_slot_filling(
                batch_hs,
                batch_slot_ranges,
            )

            slot_loss = 0.0
            for i in range(bs):
                slot_logits = batch_slot_logits[i]
                slot_labels = batch_slot_labels[i]

                dom_id = batch_domains[i]
                dom_mask = self.domain_slot_masks[dom_id].to(batch_hs.device)
                dom_mask = dom_mask.reshape((1, -1))
                slot_logits = slot_logits * dom_mask - 1e8 * (1.0 - dom_mask)

                slot_loss += F.cross_entropy(slot_logits, slot_labels)
            slot_loss /= bs

            loss = tag_loss + slot_loss

            optimizer.zero_grad()
            loss.backward()

            # clip gradient norm
            nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_max_norm, norm_type=2
            )

            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return loss

    def inference(self, model, loader, args):
        model.eval()

        all_real_tags = []
        all_pred_tags = []
        with torch.no_grad():
            for batch in tqdm(loader):
                if args.cuda:
                    batch = [elem.cuda() for elem in batch]

                batch_indices = batch[0]
                # batch_bioslot_tsg_ids = batch[1]
                # batch_tag_labels = batch[2]
                batch_slot_labels = batch[3]
                batch_slot_ranges = batch[4].detach().cpu().numpy()
                batch_slot_nums = batch[5].detach().cpu().numpy()
                batch_domains = batch[6].detach().cpu().numpy()
                batch_lens = batch[7].detach().cpu().numpy().reshape(-1)

                batch_slot_labels = [
                    batch_slot_labels[i][0:j] for i, j in enumerate(
                        batch_slot_nums
                    )
                ]
                batch_slot_ranges = [
                    batch_slot_ranges[i][0:j] for i, j in enumerate(
                        batch_slot_nums
                    )
                ]

                bs = len(batch_indices)

                # tagging
                batch_hs, batch_tag_logits = model.forward_tagging(
                    batch_indices
                )

                bt_logits = batch_tag_logits.detach()

                if self.args.use_crf:
                    pred_batch_tag_labels = model.crf.forward(bt_logits)
                else:
                    pred_batch_tag_labels = torch.argmax(bt_logits, dim=-1)
                    pred_batch_tag_labels = pred_batch_tag_labels.cpu().numpy()

                pred_batch_slot_ranges = model.decode_tags(
                    pred_batch_tag_labels
                )

                # slot filling
                pred_batch_slot_logits = model.forward_slot_filling(
                    batch_hs,
                    pred_batch_slot_ranges,
                )
                pred_batch_slot_labels = []

                for b in range(bs):
                    pbs_logits = pred_batch_slot_logits[b]

                    if len(pbs_logits) <= 0:
                        pred_batch_slot_labels.append([])
                    else:
                        dom_id = batch_domains[b]
                        dom_m = self.domain_slot_masks[dom_id]
                        dom_m = dom_m.to(batch_hs.device)
                        dom_m = dom_m.reshape((1, -1))
                        pbs_logits = pbs_logits * dom_m - 1e8 * (1.0 - dom_m)

                        pred_batch_slot_labels.append(
                            np.argmax(
                                pbs_logits.detach().cpu().numpy(),
                                axis=-1
                            )
                        )

                for k in range(bs):
                    real_tags = self.get_tags(
                        ranges=batch_slot_ranges[k],
                        labels=batch_slot_labels[k],
                        L=batch_lens[k],
                    )
                    pred_tags = self.get_tags(
                        ranges=pred_batch_slot_ranges[k],
                        labels=pred_batch_slot_labels[k],
                        L=batch_lens[k],
                    )
                    all_real_tags.append(real_tags)
                    all_pred_tags.append(pred_tags)

        return all_real_tags, all_pred_tags

    def val(self, model, loader, args):
        all_real_tags, all_pred_tags = self.inference(
            model, loader, args
        )
        f1 = f1_score(all_real_tags, all_pred_tags)
        return f1

    def test(self, model, loader1, loader2, args):
        seen_real_tags, seen_pred_tags = self.inference(
            model, loader1, args
        )
        unseen_real_tags, unseen_pred_tags = self.inference(
            model, loader2, args
        )

        if len(seen_real_tags) > 0:
            seen_f1 = f1_score(seen_real_tags, seen_pred_tags)
        else:
            seen_f1 = 0.0

        if len(unseen_real_tags) > 0:
            unseen_f1 = f1_score(unseen_real_tags, unseen_pred_tags)
        else:
            unseen_f1 = 0.0

        real_tags = seen_real_tags + unseen_real_tags
        pred_tags = seen_pred_tags + unseen_pred_tags
        f1 = f1_score(real_tags, pred_tags)
        return seen_f1, unseen_f1, f1

    def get_tags(self, ranges, labels, L):
        tags = ["O"] * L

        assert len(ranges) == len(labels)
        for k in range(len(ranges)):
            label = labels[k]
            i, j = ranges[k]

            j = min(j, L - 1)

            if i >= L:
                continue

            tags[i] = "B-{}".format(label)
            for t in range(i + 1, j + 1):
                tags[t] = "I-{}".format(label)
        return tags

    def save_ckpt(self, fpath):
        # save model
        torch.save(self.model.state_dict(), fpath)
        print("Model saved in: {}".format(fpath))

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        # logs_str = format_logs(self.logs)
        # all_logs_str.extend(logs_str)
        all_logs_str.extend(self.logs)

        append_to_logs(fpath, all_logs_str)
