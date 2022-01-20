import os

import torch
import torchvision.models
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import phases
from torch_utils import utils
from torch_utils.engine import train_one_epoch, evaluate
from utils import get_dataset


def get_model_for_training(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    state_dict_path = "./output/model_state.dict.pth"
    if os.path.exists(state_dict_path):
        print(f"Loading state from {state_dict_path}...")
        model.load_state_dict(torch.load(state_dict_path))

    return model


def get_dataloader(phase: str):
    dataset = get_dataset(phase)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4,
                            collate_fn=utils.collate_fn)
    return dataloader


def train():
    dataloaders = {}
    for phase in phases:
        dataloaders[phase] = get_dataloader(phase)

    num_classes = len(dataloaders['train'].dataset.coco.dataset['categories'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_model_for_training(num_classes, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_one_epoch(model, optimizer, dataloaders['train'], device, epoch, print_freq=10)
        scheduler.step()
        evaluate(model, dataloaders['val'], device=device)

    torch.save(model, "./output/model.pth")
    torch.save(model.state_dict(), "./output/model_state.dict.pth")
    print("Done!")


if __name__ == '__main__':
    train()
