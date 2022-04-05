from tqdm import tqdm
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


class SlotFilling():
    def __init__(
        self,
        train_set,
        val_set,
        test_set1,
        test_set2,
        model,
        domain_bioslot_masks,
        bioslot2int,
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

        self.domain_bioslot_masks = torch.FloatTensor(
            domain_bioslot_masks
        )
        self.bioslot2int = bioslot2int
        self.int2bioslot = {i: bst for bst, i in bioslot2int.items()}
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

        for batch in tqdm(loader):
            if args.cuda:
                batch = [elem.cuda() for elem in batch]

            batch_indices = batch[0]
            batch_tag_labels = batch[1]
            # batch_tag_labels = batch[2]
            # batch_slot_labels = batch[3]
            # batch_slot_ranges = batch[4].detach().cpu().numpy()
            # batch_slot_nums = batch[5].detach().cpu().numpy()
            batch_domains = batch[6].detach().cpu().numpy()
            batch_lens = batch[7].detach().cpu().numpy().reshape(-1)

            # tagging
            _, batch_tag_logits = model.forward(batch_indices)

            mask_tag_logits = []

            bs = len(batch_tag_logits)
            for i in range(bs):
                dom_id = batch_domains[i]
                dom_mask = self.domain_bioslot_masks[dom_id]
                dom_mask = dom_mask.to(batch_tag_logits.device)
                dom_mask = dom_mask.reshape((1, -1))

                tag_logits = batch_tag_logits[i]
                mask_logits = tag_logits * dom_mask - 1e8 * (1.0 - dom_mask)
                mask_tag_logits.append(mask_logits)
            mask_tag_logits = torch.stack(mask_tag_logits, dim=0)

            if self.args.use_crf:
                loss = model.crf.loss(mask_tag_logits, batch_tag_labels)
            else:
                loss = 0.0
                for i in range(bs):
                    tag_logits = mask_tag_logits[i]
                    tag_labels = batch_tag_labels[i]

                    L = batch_lens[i]
                    loss += F.cross_entropy(tag_logits[0:L], tag_labels[0:L])

                loss = loss / bs

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

        all_real_labels = []
        all_pred_labels = []
        with torch.no_grad():
            for batch in tqdm(loader):
                if args.cuda:
                    batch = [elem.cuda() for elem in batch]

                batch_indices = batch[0]
                batch_tag_labels = batch[1].detach().cpu().numpy()
                # batch_tag_labels = batch[2]
                # batch_slot_labels = batch[3]
                # batch_slot_ranges = batch[4].detach().cpu().numpy()
                # batch_slot_nums = batch[5].detach().cpu().numpy()
                batch_domains = batch[6].detach().cpu().numpy()
                batch_lens = batch[7].detach().cpu().numpy().reshape(-1)

                bs = len(batch_indices)

                # tagging
                _, batch_tag_logits = model.forward(batch_indices)

                mask_tag_logits = []

                bs = len(batch_tag_logits)
                for i in range(bs):
                    dom_id = batch_domains[i]
                    dom_m = self.domain_bioslot_masks[dom_id]
                    dom_m = dom_m.to(batch_tag_logits.device)
                    dom_m = dom_m.reshape((1, -1))

                    tag_logits = batch_tag_logits[i]
                    mask_logits = tag_logits * dom_m - 1e8 * (1.0 - dom_m)
                    mask_tag_logits.append(mask_logits)
                mask_tag_logits = torch.stack(mask_tag_logits, dim=0)

                if self.args.use_crf:
                    pred_tag_labels = model.crf.forward(mask_tag_logits)
                else:
                    pred_tag_labels = torch.argmax(
                        mask_tag_logits, dim=-1
                    )

                pred_tag_labels = pred_tag_labels.detach().cpu().numpy()

                real_labels = [
                    ts[0:L] for L, ts in zip(batch_lens, batch_tag_labels)
                ]
                pred_labels = [
                    ts[0:L] for L, ts in zip(batch_lens, pred_tag_labels)
                ]

                all_real_labels.extend(real_labels)
                all_pred_labels.extend(pred_labels)

        real_labels = [
            [self.int2bioslot[i] for i in ts] for ts in all_real_labels
        ]
        pred_labels = [
            [self.int2bioslot[i] for i in ts] for ts in all_pred_labels
        ]

        return real_labels, pred_labels

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
