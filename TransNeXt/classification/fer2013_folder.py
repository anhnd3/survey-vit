# classification/fer2013_folder.py
from torchvision import transforms, datasets
from torch.utils.data import random_split

def build_datasets(root: str,
                  img_size: int = 224,
                  is_train: bool = True):
    """
    Build FER-2013 ImageFolder datasets.

    Parameters
    ----------
    root : str
        Path to the folder that contains `train/` and `test/`.
    img_size : int
        Final square resolution (must match --input-size).
    is_train : bool
        True  -> return train & val split  (90 / 10)
        False -> return test set only.

    Returns
    -------
    dataset          : torch.utils.data.Dataset
    dataset_val/test : Dataset or None
    num_classes      : int  (always 7)
    """
    # ---------- transforms ----------
    tfms_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2, 15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    tfms_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if is_train:
        # full training set
        full = datasets.ImageFolder(f"{root}/train", transform=tfms_train)
        val_len = int(0.1 * len(full))            # 10 % for validation
        train_len = len(full) - val_len
        train_set, val_set = random_split(full, [train_len, val_len])
        # use evaluation transforms for validation subset
        val_set.dataset.transform = tfms_eval
        return train_set, val_set, 7
    else:
        test_set = datasets.ImageFolder(f"{root}/test", transform=tfms_eval)
        return test_set, None, 7
