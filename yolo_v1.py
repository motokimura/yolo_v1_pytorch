import torch
import torch.nn as nn
import torch.nn.functional as F

from darknet import DarkNet
from util_layers import Flatten


class YOLOv1(nn.Module):
    def __init__(self, features, num_classes=20, num_bboxes=2, bn=True):
        super(YOLOv1, self).__init__()

        self.feature_size = 7
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes

        self.features = features
        self.yolo = self._make_yolo_layers(bn)

    def _make_yolo_layers(self, bn):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        if bn:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),

                Flatten(),
                nn.Linear(7 * 7 * 1024, 4096),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5, inplace=True),
                nn.Linear(4096, S * S * (5 * B + C))
            )

        else:
            net = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),

                Flatten(),
                nn.Linear(7 * 7 * 1024, 4096),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5, inplace=True), # should be replaced or used together with BatchNorm?
                nn.Linear(4096, S * S * (5 * B + C))
            )

        return net

    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        x = self.features(x)
        x = self.yolo(x)
        x = torch.Sigmoid(x)
        x = x.view(-1, S, S, 5 * B + C)
        return x


def test():
    from torch.autograd import Variable

    # Build model with randomly initialized weights
    darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
    yolo = YOLOv1(darknet)

    # Prepare a dummy image to input
    image = torch.rand(1, 3, 448, 448)
    image = Variable(image)

    # Forward
    output = yolo(image)
    # Check ouput tensor size, which should be [1, 7, 7, 30]
    print(output.size())


if __name__ == '__main__':
    test()
