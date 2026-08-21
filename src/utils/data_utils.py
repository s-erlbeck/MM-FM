import torch
import numpy as np


class ClassBalancedSubset(torch.utils.data.Dataset):
    """
    Wrapper that samples a percentage of data from each class while maintaining class balance,
    optionally also subsampling the set of classes itself.
    Only supports ImageFolder (requires .targets attribute).

    Args:
        base_dataset: The underlying dataset (must have .targets attribute)
        sample_percentage: Fraction of samples to use per (kept) class (0.0-1.0)
        class_percentage: Fraction of classes to keep (0.0-1.0)
        seed: Random seed for reproducible sampling
    """
    def __init__(self, base_dataset, sample_percentage=1.0, class_percentage=1.0, seed=42):
        self.base_dataset = base_dataset
        self.sample_percentage = sample_percentage
        self.class_percentage = class_percentage

        # Only ImageFolder is supported
        if not hasattr(base_dataset, 'targets'):
            raise ValueError("ClassBalancedSubset only supports ImageFolder datasets")

        all_labels = base_dataset.targets

        # Build per-class index lists
        class_to_indices = {}
        for idx, label in enumerate(all_labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        rng = np.random.RandomState(seed)

        # Subsample the classes themselves
        all_class_ids = sorted(class_to_indices.keys())
        if class_percentage < 1.0:
            n_classes = max(1, int(len(all_class_ids) * class_percentage))
            sampled = rng.choice(all_class_ids, size=n_classes, replace=False)
            self.selected_classes = sorted(sampled.tolist())
        else:
            self.selected_classes = all_class_ids

        # Sample indices per kept class
        self.selected_indices = []

        for class_id in self.selected_classes:
            indices = class_to_indices[class_id]

            # Sample
            n_samples = max(1, int(len(indices) * sample_percentage))  # At least 1 sample per class
            sampled = rng.choice(indices, size=n_samples, replace=False)
            self.selected_indices.extend(sampled)

        self.selected_indices = sorted(self.selected_indices)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        return self.base_dataset[original_idx]
