import os
from typing import Optional, Callable, Tuple, Any

import torch.utils.data
from torchvision.datasets import CocoDetection


class ChessPiecesDataset(CocoDetection):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        ann_file = os.path.join(root, "_annotations.coco.json")

        super().__init__(root, ann_file, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, targets = super().__getitem__(index)

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for target in targets:
            bbox = target['bbox']
            bbox_new_format = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

            boxes.append(bbox_new_format)
            labels.append(target['category_id'])
            area.append(target['area'])
            iscrowd.append(target['iscrowd'])
        targets = {
            'boxes': torch.tensor(boxes),
            'labels': torch.tensor(labels),
            'image_id': torch.tensor(targets[0]['image_id']),
            'area': torch.tensor(area),
            'iscrowd': torch.tensor(iscrowd),
        }

        return image, targets
