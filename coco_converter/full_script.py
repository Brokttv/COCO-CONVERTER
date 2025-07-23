import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


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
            image_id = annot.get("image_id", None)
            if image_id is not None:
                self.id_annots_pairs.setdefault(image_id, []).append(annot)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data_info = self.images[idx]
        image_id = data_info["id"]
        file_name = data_info["file_name"]
        
        image_path = os.path.join(self.img_dir, file_name)
        image = read_image(image_path)
        
        annotations = self.id_annots_pairs.get(image_id, [])
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            annotations = self.target_transform(annotations)
        
        return image, annotations


class Convert_to_COCO:
    def process_folder(self, data_dir: str) -> Dict:
        data_dir = Path(data_dir)
        images = []
        annotations = []
        category_map = {}

        if not data_dir.exists() or not data_dir.is_dir():
            raise NotADirectoryError(f"Folder not found: {data_dir}")

        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                label = class_dir.name
                if label not in category_map:
                    category_map[label] = len(category_map)

                for image_path in class_dir.glob("*"):
                    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                        continue

                    img_id = len(images)
                    rel_path = image_path.relative_to(data_dir)

                    images.append({
                        "id": img_id,
                        "file_name": str(rel_path),
                        "width": None,
                        "height": None
                    })
                    annotations.append({
                        "image_id": img_id,
                        "category_id": category_map[label],
                        "id": len(annotations)
                    })

        categories = [
            {"id": cat_id, "name": cat_name}
            for cat_name, cat_id in category_map.items()
        ]
        
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

    def process_csv(self, csv_path: str) -> Dict:
        csv_path = Path(csv_path)
        images = []
        annotations = []
        category_map = {}

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Issues reading CSV: {e}")

        for idx, row in df.iterrows():
            image_path = Path(row["image_path"])
            label = str(row["label"])
            
            if label not in category_map:
                category_map[label] = len(category_map)

            images.append({
                "id": idx,
                "file_name": str(image_path),
                "width": None,
                "height": None
            })
            annotations.append({
                "image_id": idx,
                "category_id": category_map[label],
                "id": len(annotations)
            })

        categories = [
            {"id": cat_id, "name": cat_name}
            for cat_name, cat_id in category_map.items()
        ]
        
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

    def process_flat_folder(self,data_dir:str):
      data_dir

    def data_type_detection(self, data_dir: str) -> Tuple[Dict, Path]:
        data_dir = Path(data_dir)

        if data_dir.is_file() and data_dir.suffix.lower() == ".csv":
            print("Processing CSV dataset")
            data = self.process_csv(data_dir)
            img_dir = data_dir.parent
        elif data_dir.is_dir():
            print("Processing Folder dataset")
            data = self.process_folder(data_dir)
            img_dir = data_dir
        else:
            raise ValueError(f"Unsupported input type: {data_dir} - should be either a folder or .csv file!")

        return data, img_dir

    def yolo_inference(self, source: str, output_dir: str = "yolo_out") -> Path:
        cmd = (f"python detect.py --weights yolov5s.pt --source {source} "
               f"--save-txt --save-conf --project {output_dir} --name detected")
        
        if os.system(cmd) != 0:
            raise RuntimeError("YOLO detection failed!")
        return Path(output_dir) / "detected" / "labels"

    def coco_format(self, input_data: Dict, labels_dir: str, images_dir: str) -> Dict:
        labels_dir = Path(labels_dir)
        images_dir = Path(images_dir)
        
        if not labels_dir.exists() or not labels_dir.is_dir():
            raise NotADirectoryError(f"{labels_dir} is not a directory")
        if not images_dir.exists() or not images_dir.is_dir():
            raise NotADirectoryError(f"{images_dir} is not a directory")

        image_map = {img["file_name"]: img["id"] for img in input_data["images"]}
        new_annotations = []

        for label_file in labels_dir.glob("*.txt"):
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name

            if not image_path.exists():
                print(f"{image_path} does not exist; skipping...")
                continue

            width, height = Image.open(image_path).size

            with open(label_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) == 5:
                    class_id, x_c, y_c, w, h = map(float, parts)
                    conf = None
                elif len(parts) == 6:
                    class_id, x_c, y_c, w, h, conf = map(float, parts)
                else:
                    continue

                x_min = (x_c - w/2) * width
                y_min = (y_c - h/2) * height
                bbox = [x_min, y_min, w * width, h * height]

                annotation = {
                    "id": len(new_annotations),
                    "image_id": image_map[image_name],
                    "bbox": bbox,
                    "category_id": int(class_id),
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0
                }
                
                if conf is not None:
                    annotation["conf"] = conf
                
                new_annotations.append(annotation)

        input_data["annotations"] = new_annotations
        return input_data


def main():
    parser = argparse.ArgumentParser(description="Convert folder or csv to COCO format with YOLO detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to input data (folder or csv file)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path for COCO format")
    
    args = parser.parse_args()
    converter = Convert_to_COCO()

    try:
        coco_data, img_dir = converter.data_type_detection(args.data_dir)
        labels_dir = converter.yolo_inference(str(img_dir))
        coco_data = converter.coco_format(coco_data, labels_dir, img_dir)

        with open(args.output, "w") as f:
            json.dump(coco_data, f, indent=2)

        dataset = CocoDataset(args.output, img_dir)
        print(f"Created dataset with {len(dataset)} items")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()




    




  
