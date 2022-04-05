import torch
from torch.utils import data


class SnipsDataset(data.Dataset):
    def __init__(self, indices, tags, slots, slot_ranges, slot_masks):
        self.indices = indices
        self.tags = tags
        self.slots = slots
        self.slot_ranges = slot_ranges
        self.slot_masks = slot_masks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        indices = self.indices[k]
        tags = self.tags[k]
        slots = self.slots[k]
        slot_ranges = self.slot_ranges[k]
        slot_masks = self.slot_masks[k]

        indices = torch.LongTensor(indices)
        tags = torch.LongTensor(tags)
        slots = torch.LongTensor(slots)
        slot_ranges = torch.LongTensor(slot_ranges)
        slot_masks = torch.FloatTensor(slot_masks)
        return indices, tags, slots, slot_ranges, slot_masks
