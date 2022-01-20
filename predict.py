import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

plt.rcParams["savefig.bbox"] = 'tight'


def show_images(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    f = plt.figure(figsize=(60, 30))

    n = len(imgs)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        plt.subplot(1, n, i + 1)
        plt.imshow(np.asarray(img))
    plt.show()


def predict(images):
    model = torch.load("./output/model.pth")
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output = model([image.to(device) / 255 for image in images])
    results = []

    for i in range(len(output)):
        boxes = output[i]['boxes']
        scores = output[i]['scores']
        keep = torchvision.ops.nms(boxes, scores, 0.3)
        labels = [str(l) for l in output[i]['labels'][keep].tolist()]
        results.append(draw_bounding_boxes(images[i], boxes[keep], labels=labels, colors="blue", width=5,
                                           font_size=16))
    show_images(results)


if __name__ == "__main__":
    images = ["./data/chess_pieces/test/a3863d0be6002c21b20ac88817b2c56f_jpg.rf.e421134b139d57e02e7df9468a35c1fb.jpg",
              "./data/chess_pieces/test/f1a24b6bb778ee11ba33687415aa84f2_jpg.rf.6e35192bbbb13f887540067e07d5d660.jpg"]
    predict(images)
