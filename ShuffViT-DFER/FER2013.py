from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class FER2013(Dataset):
    def __init__(self, split='train', transform=None, root='datasets/FER2013'):
        self.root = Path(root) / split
        self.paths = sorted(self.root.glob('*/*.png'))
        self.labels = [int(p.parent.name) for p in self.paths]
        self.transform = transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]
