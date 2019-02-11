import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from voc import VOCDataset
from darknet import DarkNet
from yolo_v1 import YOLOv1
from loss import Loss

import os
import numpy as np
import math
from datetime import datetime

from tensorboardX import SummaryWriter


# Check if GPU devices are available.
use_gpu = torch.cuda.is_available()
print('use_gpu: {}'.format(use_gpu))
print('cuda current_device: {}'.format(torch.cuda.current_device()))
print('cuda device_count: {}'.format(torch.cuda.device_count()))

# Path to data dir.
image_dir = 'data/VOC_allimgs/'

# Path to label files.
train_label = ['data/voc2007.txt', 'data/voc2012.txt']
val_label = 'data/voc2007test.txt'

# Path to checkpoint file containing pre-trained DarkNet weight.
checkpoint_path = 'weights/darknet/model_best.pth.tar'

# Frequency to print/log the results.
print_freq = 5
tb_log_freq = 5

# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64

# Learning rate scheduling.
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Load pre-trained darknet.
darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
darknet.features = torch.nn.DataParallel(darknet.features)

src_state_dict = torch.load(checkpoint_path)['state_dict']
dst_state_dict = darknet.state_dict()

for k in dst_state_dict.keys():
    print('Loading weight of', k)
    dst_state_dict[k] = src_state_dict[k]
darknet.load_state_dict(dst_state_dict)

# Load YOLO model.
yolo = YOLOv1(darknet.features)
yolo.conv_layers = torch.nn.DataParallel(yolo.conv_layers)

if use_gpu:
    yolo.cuda()

# Setup loss and optimizer.
criterion = Loss(feature_size=yolo.feature_size)
optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

# Load Pascal-VOC dataset.
train_dataset = VOCDataset(True, image_dir, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = VOCDataset(False, image_dir, val_label)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('Number of training images: ', len(train_dataset))

# Open TensorBoardX summary writer
log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('results/yolo', log_dir)
writer = SummaryWriter(log_dir=log_dir)

# Training loop.
logfile = open(os.path.join(log_dir, 'log.txt'), 'w')
best_val_loss = np.inf

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    yolo.train()
    total_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(train_loader):
        # Update learning rate.
        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        lr = get_lr(optimizer)

        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        if use_gpu:
            imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute loss.
        preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item()
        total_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter

        # Backward to update model weight.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss.
        if i % print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
            % (epoch, num_epochs, i, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

        # TensorBoard.
        n_iter = epoch * len(train_loader) + i
        if n_iter % tb_log_freq == 0:
            writer.add_scalar('train/loss', loss_this_iter, n_iter)
            writer.add_scalar('lr', lr, n_iter)

    # Validation.
    yolo.eval()
    val_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(val_loader):
        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        if use_gpu:
            imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute validation loss.
        with torch.no_grad():
            preds = yolo(imgs)
        loss = criterion(preds, targets)
        loss_this_iter = loss.item()
        val_loss += loss_this_iter * batch_size_this_iter
        total_batch += batch_size_this_iter
    val_loss /= float(total_batch)

    # Save results.
    logfile.writelines(str(epoch + 1) + '\t' + str(val_loss) + '\n')
    logfile.flush()
    torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_latest.pth'))
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))

    # Print.
    print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f'
    % (epoch + 1, num_epochs, val_loss, best_val_loss))

    # TensorBoard.
    writer.add_scalar('test/loss', val_loss, epoch + 1)

write.close()
logfile.close()
