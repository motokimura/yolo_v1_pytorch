import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from voc import VOCDataset
from darknet import DarkNet
from yolo_v1 import YOLOv1
from loss import Loss

import numpy as np

# Check if GPU devices are available.
use_gpu = torch.cuda.is_available()
print('use_gpu: {}'.format(use_gpu))
print('cuda device, device count: ', torch.cuda.current_device(), torch.cuda.device_count())

# Path to data dir.
image_dir = 'data/VOC_allimgs/'

# Path to label files.
train_label = ['data/voc2007.txt', 'data/voc2012.txt']
val_label = 'data/voc2007test.txt'

# Path to checkpoint file containing pre-trained DarkNet weight.
checkpoint_path = 'models/ckpt_darknet_bn.pth.tar'

# Hyper parameters.
initial_lr = 1.0e-5
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64

# Learning rate scheduling.
def get_lr(epoch, current_lr):
    if epoch == 0:
        lr = initial_lr
    elif epoch == 1:
        lr = 0.0001
    elif epoch == 2:
        lr = 0.0005
    elif epoch == 3:
        lr = 0.001
    elif epoch == 4:
        lr = 0.005
    elif epoch == 5:
        lr = 0.01
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        lr = current_lr

    return lr

# Load pre-trained darknet.
darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
darknet.features = torch.nn.DataParallel(darknet.features)

src_state_dict = torch.load(checkpoint_path)['state_dict']
dst_state_dict = darknet.state_dict()

for k in dst_state_dict.keys():
    print(k)
    dst_state_dict[k] = src_state_dict[k]
darknet.load_state_dict(dst_state_dict)

# Load YOLO model.
yolo = YOLOv1(darknet.features)
yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layers)

if use_gpu:
    yolo.cuda()

# Setup loss and optimizer.
criterion = Loss(feature_size=yolo.feature_size)
optimizer = torch.optim.SGD(yolo.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

# Load Pascal-VOC dataset.
train_dataset = VOCDataset(True, image_dir, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = VOCDataset(False, image_dir, val_label)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Number of training images: ', len(train_dataset))

# Training loop.
logfile = open('log.txt', 'w')

best_val_loss = np.inf
lr = initial_lr

for epoch in range(num_epochs):

    # Schedule learning rate.
    lr = get_lr(epoch, lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))
    print('Learning rate for this epoch: {}'.format(lr))

    # Training.
    yolo.train()
    total_loss = 0.0

    for i, (imgs, targets) in enumerate(train_loader):
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        if use_gpu:
            imgs, targets = imgs.cuda(), targets.cuda()

        preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item() / float(batch_size_this_iter)
        total_loss += loss_this_iter

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
            % (epoch, num_epochs, i, len(train_loader), loss_this_iter, total_loss / (i+1)))

    # Validation.
    yolo.eval()
    val_loss = 0.0

    for i, (imgs, targets) in enumerate(val_loader):
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        if use_gpu:
            imgs, targets = imgs.cuda(), targets.cuda()

        with torch.no_grad():
            preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item() / float(batch_size_this_iter)
        val_loss += loss_this_iter
    val_loss /= float(len(val_loader))

    # Save results.
    logfile.writelines(str(epoch + 1) + '\t' + str(val_loss) + '\n')
    logfile.flush()
    torch.save(yolo.state_dict(),'yolo.pth')

    if best_val_loss > val_loss:
        best_val_loss = val_loss
        torch.save(yolo.state_dict(), 'yolo_best.pth')

    print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
    % (epoch + 1, num_epochs, val_loss, best_val_loss))

logfile.close()
