import os

import fiftyone as fo
from torchvision import transforms

from config import datasets_dir
from datasets.chess_pieces_dataset import ChessPiecesDataset


def get_dataset(phase):
    return ChessPiecesDataset(os.path.join(datasets_dir, phase), transforms.ToTensor())


def explore_dataset(phase):
    coco_dataset = fo.Dataset.from_dir(
        name=f"chess_pieces_{phase}",
        dataset_type=fo.types.COCODetectionDataset,
        data_path=f"./data/chess_pieces/{phase}",
        labels_path=f"./data/chess_pieces/{phase}/_annotations.coco.json",
        include_id=True)

    session = fo.launch_app(coco_dataset, port=5151)
    session.wait()
