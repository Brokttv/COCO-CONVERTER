import argparse
import json
import sys
from converter import Convert_to_COCO
from dataset import CocoDataset

def main():
    parser = argparse.ArgumentParser(description="Convert folder or CSV to COCO format + YOLO labels")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder or CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")

    args = parser.parse_args()
    converter = Convert_to_COCO()

    try:
        coco_data, img_dir = converter.data_type_detection(args.data_dir)
        labels_dir = converter.yolo_inference(str(img_dir))
        coco_data = converter.coco_format(coco_data, labels_dir, img_dir)

        with open(args.output, "w") as f:
            json.dump(coco_data, f, indent=2)

        dataset = CocoDataset(args.output, img_dir)
        print(f"Created dataset with {len(dataset)} items.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
