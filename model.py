import torch
import torch.nn as nn

from crf import CRF


class SlotFillingModel(nn.Module):

    def __init__(self, label_embeds, args):
        super().__init__()
        self.label_embeds = torch.FloatTensor(label_embeds)
        self.args = args

        self.encoder = nn.Embedding(args.n_vocab, args.w_dim)
        self.dropout = nn.Dropout(0.3, inplace=True)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=args.w_dim,
            hidden_size=args.hidden_size,
            dropout=0.3,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        fc_size = 2 * args.hidden_size

        # BIO classifier
        self.tag_classifier = nn.Linear(fc_size, args.n_bioslots)

        # crf
        if args.use_crf:
            self.crf = CRF(args.n_bioslots)

    def forward(self, batch_indices):
        batch_embeds = self.encoder(batch_indices).detach()
        batch_embeds = self.dropout(batch_embeds)

        # LSTM
        bs, seq_len = batch_embeds.shape[0], batch_embeds.shape[1]
        h0 = torch.zeros(4, bs, self.args.hidden_size)
        c0 = torch.zeros(4, bs, self.args.hidden_size)
        h0 = h0.to(batch_indices.device)
        c0 = c0.to(batch_indices.device)
        batch_hs, _ = self.lstm(batch_embeds, (h0, c0))
        batch_hs = batch_hs.view(bs, seq_len, -1)

        # tag classifier
        if self.args.clf == "clf":
            batch_tag_logits = self.tag_classifier(batch_hs)
        else:
            label_embeds = self.label_embeds.to(batch_hs.device)

            bs, seq_len = batch_hs.shape[0], batch_hs.shape[1]
            batch_tag_logits = batch_hs.reshape((bs * seq_len, -1)).mm(
                label_embeds.transpose(0, 1)
            ).reshape((bs, seq_len, -1))
        return batch_hs, batch_tag_logits

    def load_w2v(self, embeddings):
        """ Load word embeddings
        """
        if embeddings is not None:
            weights = torch.FloatTensor(embeddings)
            self.encoder = nn.Embedding.from_pretrained(weights)


class SlotFillingCoachModel(nn.Module):

    def __init__(self, label_embeds, args):
        super().__init__()
        self.label_embeds = torch.FloatTensor(label_embeds)
        self.args = args

        self.encoder = nn.Embedding(args.n_vocab, args.w_dim)
        self.dropout = nn.Dropout(0.3, inplace=True)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=args.w_dim,
            hidden_size=args.hidden_size,
            dropout=0.3,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        fc_size = 2 * args.hidden_size

        # BIO classifier
        self.tag_classifier = nn.Linear(fc_size, args.n_tags)

        # crf
        if args.use_crf:
            self.crf = CRF(args.n_tags)

        if args.pool == "lstm":
            self.slot_lstm = nn.LSTM(
                input_size=args.hidden_size * 2,
                hidden_size=args.hidden_size,
                bidirectional=True,
                batch_first=True,
            )

        # Slot classifier
        if args.clf == "clf":
            self.slot_classifier = nn.Linear(fc_size, args.n_slots)
        else:
            pass

    def forward_tagging(self, batch_indices):
        batch_embeds = self.encoder(batch_indices).detach()
        batch_embeds = self.dropout(batch_embeds)

        # GRU
        bs, seq_len = batch_embeds.shape[0], batch_embeds.shape[1]
        h0 = torch.zeros(4, bs, self.args.hidden_size)
        h0 = h0.to(batch_indices.device)
        c0 = torch.zeros(4, bs, self.args.hidden_size)
        c0 = c0.to(batch_indices.device)

        batch_hs, _ = self.lstm(batch_embeds, (h0, c0))
        batch_hs = batch_hs.view(bs, seq_len, -1)

        # tag classifier
        batch_tag_logits = self.tag_classifier(batch_hs)
        return batch_hs, batch_tag_logits

    def forward_slot_filling_single(self, hs, slot_ranges):
        """ single sentence
            hs: torch.Tensor, shape = (len, d)
            slot_ranges: list, list of [(bi, ei), ...]
        """
        assert len(hs.shape) == 2

        pools = []
        if self.args.pool == "avg":
            for n, (i, j) in enumerate(slot_ranges):
                slot_pool = hs[i:j + 1].mean(dim=0)
                pools.append(slot_pool)
        elif self.args.pool == "sum":
            for n, (i, j) in enumerate(slot_ranges):
                slot_pool = hs[i:j + 1].sum(dim=0)
                pools.append(slot_pool)
        elif self.args.pool == "max":
            for n, (i, j) in enumerate(slot_ranges):
                slot_pool, _ = torch.max(hs[i:j + 1], dim=0)
                pools.append(slot_pool)
        elif self.args.pool == "lstm":
            h0 = torch.zeros(2, 1, self.args.hidden_size)
            h0 = h0.to(hs.device)
            c0 = torch.zeros(2, 1, self.args.hidden_size)
            c0 = c0.to(hs.device)

            for n, (i, j) in enumerate(slot_ranges):
                slot_hs = hs[i:j + 1].unsqueeze(dim=0)
                slot_pool, _ = self.slot_lstm(slot_hs, (h0, c0))
                slot_pool = slot_pool.sum(dim=1).squeeze()
                pools.append(slot_pool)
        else:
            raise ValueError("No such pool: {}".format(self.args.pool))
        pools = torch.stack(pools, dim=0)

        # slot classifier
        if self.args.clf == "clf":
            slot_logits = self.slot_classifier(pools)
        else:
            label_embeds = self.label_embeds.to(pools.device)
            slot_logits = pools.mm(label_embeds.transpose(0, 1))
        return slot_logits

    def forward_slot_filling(self, batch_hs, batch_slot_ranges):
        """ hs.shape = (bs, len, d)
            slot_masks.shape = (bs, max_n_slot, len)
        """
        batch_slot_logits = []
        for hs, slot_ranges in zip(batch_hs, batch_slot_ranges):
            if len(slot_ranges) <= 0:
                slot_logits = torch.FloatTensor([])
            else:
                slot_logits = self.forward_slot_filling_single(
                    hs, slot_ranges
                )
            batch_slot_logits.append(slot_logits)
        return batch_slot_logits

    def decode_tags(self, batch_tag_labels):
        """ tag_logits: (bs, len, n_tags)
        """
        # B I 0: 1, 2, 0
        batch_ranges = []

        for b in range(len(batch_tag_labels)):
            tags = batch_tag_labels[b]
            ranges = []
            bi = -1
            ei = -1
            for i, tag in enumerate(tags):
                if tag == 1:
                    if bi == -1:
                        bi = i
                        ei = i
                    else:
                        ranges.append([bi, ei])
                        bi = i
                        ei = i
                elif tag == 2:
                    if bi != -1:
                        ei = i
                elif tag == 0:
                    if bi != -1:
                        ranges.append([bi, ei])
                    bi = -1
                    ei = -1

            if bi != -1:
                ranges.append([bi, ei])

            batch_ranges.append(ranges)
        return batch_ranges

    def load_w2v(self, embeddings):
        """ Load word embeddings
        """
        if embeddings is not None:
            weights = torch.FloatTensor(embeddings)
            self.encoder = nn.Embedding.from_pretrained(weights)
