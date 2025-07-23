
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json

class CocoDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        with open(json_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.id_annots_pairs = {}
        for annot in data.get("annotations", []):
            image_id = annot.get("image_id")
            if image_id is not None:
                self.id_annots_pairs.setdefault(image_id, []).append(annot)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = info["id"]
        image_path = os.path.join(self.img_dir, info["file_name"])
        image = read_image(image_path)
        annotations = self.id_annots_pairs.get(image_id, [])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            annotations = self.target_transform(annotations)

        return image, annotations

