import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = 0 if "no_pothole" in img_path else 1  # basic label example
        if self.transform:
            image = self.transform(image)
        return image, label
