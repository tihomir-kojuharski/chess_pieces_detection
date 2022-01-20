import math
import os
from collections import defaultdict
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from config import phases, datasets_dir
from datasets.chess_pieces_dataset import ChessPiecesDataset
import seaborn as sns

from utils import get_dataset


class PositionHeatmap:
    CELL_SIZE = 30  # px
    IMAGES_WIDTH = 2048
    IMAGES_HEIGHT = 1371

    def __init__(self):
        heatmap_shape = self.__get_grid_cell_position((self.IMAGES_HEIGHT, self.IMAGES_WIDTH))
        self.__heatmap = np.zeros((heatmap_shape[0] + 1, heatmap_shape[1] + 1))

    def __get_grid_cell_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        return (math.ceil(position[0] / self.CELL_SIZE) - 1,
                math.ceil(position[1] / self.CELL_SIZE) - 1)

    def add(self, position: Tuple[int, int]):
        cell = self.__get_grid_cell_position(position)

        self.__heatmap[cell[0], cell[1]] += 1

    def get_distribution(self) -> np.array:
        return self.__heatmap


def plot_position_distributions():
    for phase_id, phase in enumerate(phases):
        dataset = get_dataset(phase)

        position_heatmaps = defaultdict(PositionHeatmap)

        for image, target in dataset:
            for i in range(len(target["boxes"])):
                bbox = target["boxes"][i]
                label = target["labels"][i].item()

                bbox_center = int((bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2)

                position_heatmaps[label].add(bbox_center)
        heatmaps_dir = f"""output/heatmaps/"""
        os.makedirs(heatmaps_dir, exist_ok=True)

        for category in dataset.coco.cats.values():
            plt.figure()
            sns.heatmap(position_heatmaps[category["id"]].get_distribution()) \
                .set_title(f"""{category["name"]} - {phase}""")
            plt.savefig(f"""{heatmaps_dir}/{category["name"]}_{phase_id + 1}_{phase}.jpg""")
            plt.close()


if __name__ == "__main__":
    plot_position_distributions()
