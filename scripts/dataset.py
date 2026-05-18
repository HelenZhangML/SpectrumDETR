import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class XPSSetDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, map_location="cpu")

        if not isinstance(self.data, list):
            raise ValueError(f"Expected list of examples in {pt_path}, got {type(self.data)}")

        if len(self.data) == 0:
            raise ValueError(f"Dataset is empty: {pt_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        spectrum = ex["spectrum"]
        target = ex["env_embeddings"]

        if not torch.is_tensor(spectrum):
            spectrum = torch.tensor(spectrum, dtype=torch.float32)
        else:
            spectrum = spectrum.float()

        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        else:
            target = target.float()

        if spectrum.ndim == 1:
            spectrum = spectrum.unsqueeze(0)

        out = {
            "spectra": spectrum,
            "targets": target,
        }

        for key in [
            "key", "split", "kind", "source_idx", "aug_id",
            "is_augmented", "mix_indices", "mix_weights"
        ]:
            if key in ex:
                out[key] = ex[key]

        return out


def collate_set(batch):
    spectra = torch.stack([b["spectra"] for b in batch], dim=0)
    targets = [b["targets"] for b in batch]

    out = {
        "spectra": spectra,
        "targets": targets,
    }

    meta_keys = set().union(*[set(b.keys()) for b in batch]) - {"spectra", "targets"}
    for key in meta_keys:
        out[key] = [b.get(key, None) for b in batch]

    return out


def split_by_group(dataset, seed=0, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6

    keys = [dataset[i]["key"] for i in range(len(dataset))]

    group_to_indices = defaultdict(list)
    for i, key in enumerate(keys):
        group_to_indices[key].append(i)

    groups = list(group_to_indices.keys())

    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = int(ratios[0] * n_groups)
    n_val = int(ratios[1] * n_groups)

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]

    def expand(group_list):
        idx = []
        for group in group_list:
            idx.extend(group_to_indices[group])
        return idx

    train_idx = expand(train_groups)
    val_idx = expand(val_groups)
    test_idx = expand(test_groups)

    info = {
        "seed": seed,
        "ratios": list(ratios),
        "n_groups": n_groups,
        "n_groups_train": len(train_groups),
        "n_groups_val": len(val_groups),
        "n_groups_test": len(test_groups),
        "n_examples": len(dataset),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }

    return train_idx, val_idx, test_idx, info