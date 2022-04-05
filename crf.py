import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def loss(self, feats, tags):
        if len(feats.shape) != 3:
            raise ValueError(
                "feats must be 3-d got {}-d".format(feats.shape)
            )

        if len(tags.shape) != 2:
            raise ValueError(
                'tags must be 2-d but got {}-d'.format(tags.shape)
            )

        if feats.shape[:2] != tags.shape:
            raise ValueError(
                'First two dimensions of feats and tags must match ',
                feats.shape, tags.shape
            )

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        feat_score = feats.gather(
            2, tags.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)

        tags_pairs = tags.unfold(1, 2, 1)

        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(
                self.num_tags, num_tags
            ))

        # [batch_size, num_tags]
        a = feats[:, 0] + self.start_transitions.unsqueeze(0)

        # [1, num_tags, num_tags] from -> to
        transitions = self.transitions.unsqueeze(0)

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1)
            a = self._log_sum_exp(a.unsqueeze(-1) + transitions + feat, 1)

        return self._log_sum_exp(a + self.stop_transitions.unsqueeze(0), 1)

    def _viterbi(self, feats):
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError('num_tags should be {} but got {}'.format(
                self.num_tags, num_tags))

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)
        transitions = self.transitions.unsqueeze(0)
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i]
            v, idx = (v.unsqueeze(-1) + transitions).max(1)

            paths.append(idx)
            v = (v + feat)

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)

    def _log_sum_exp(self, logits, dim):
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()
