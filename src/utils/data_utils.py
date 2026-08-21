import io
import os
from zipfile import ZipFile, BadZipfile

from PIL import Image
import numpy as np
import torch


class ImageNetDataset(torch.utils.data.Dataset):
    # compare https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/data_management/central_datasets/
    def __init__(self, zfpath: str, transform=None):
        self.zfpath = zfpath
        self.transform = transform

        # Avoid reusing the file handle created here, for known issue with multi-worker:
        # https://discuss.pytorch.org/t/dataloader-with-zipfile-failed/42795
        self.zf = None
        with ZipFile(self.zfpath) as zf:
            self.imglist: list[str] = [path for path in zf.namelist() if path.endswith(".jpg")]

        # Images are structured in directories based on class
        with open(os.path.join(os.path.dirname(self.zfpath), "devkit", "data", "map_clsloc.txt")) as f:
            self.classes: dict[str, int] = dict(self.parse_row(row) for row in f)

        # populate targets in order to use ClassBalancedSubset
        self.targets = [self.get_label(p) for p in self.imglist]

    def parse_row(self, row: str) -> tuple[str, int]:
        classname, classnum, _ = row.split()
        return classname, (int(classnum) - 1)

    def get_label(self, path: str) -> int:
        classname: str = path.split("/")[-2]
        return self.classes[classname]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        if self.zf is None:
            self.zf=ZipFile(self.zfpath)

        imgpath = self.imglist[idx]
        try:
            img = Image.open(io.BytesIO(self.zf.read(imgpath)))
        except BadZipfile:
            # It seems that sometimes the zipfile handle can become bad
            self.zf = ZipFile(self.zfpath)
            img = Image.open(io.BytesIO(self.zf.read(imgpath)))
        if self.transform is not None:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label



class ClassBalancedSubset(torch.utils.data.Dataset):
    """
    Wrapper that samples a percentage of data from each class while maintaining class balance,
    optionally also subsampling the set of classes itself.
    Requires the base dataset to have a .targets attribute (e.g. ImageFolder, ImageNetDataset).

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

        if not hasattr(base_dataset, 'targets'):
            raise ValueError("ClassBalancedSubset requires a base dataset with a .targets attribute")

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
