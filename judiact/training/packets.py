"""One planned packet per rank and update."""

from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, packets):
        self.packets = packets

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, index):
        return self.packets[index]


def packet_collator(packets):
    assert len(packets) == 1
    return packets[0]
