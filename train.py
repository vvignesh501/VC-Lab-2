import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

def train_net(net,
              epochs=5,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              gpu=False):
    loader = DataLoader(data_dir)

    N_train = loader.n_train()

    # Loss function
    criterion = nn.CrossEntropyLoss()
 
    optimizer = optim.SGD(net.parameters(),
                            lr=lr,
                            momentum=0.99,
                            weight_decay=0.0005)


    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            img_tensor = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()  # Convert numpy array to tensor
            shape1=img_tensor.shape
            mask_tensor=label.shape
            #mask_tensor = torch.from_numpy(label.reshape(1,label_shape[0],label_shape[1])).long()
            #mask_tensor=mask_tensor.cpu().detach().numpy().squeeze(0)
            #mask_tensor=[t.numpy() for t in mask_tensor]
            #shape2=mask_tensor.shape
            # todo: load image tensor to gpu
            #if gpu:

            # todo: get prediction and getLoss()
            images=Variable(img_tensor)
            #mask=Variable(label)


            pred_img=net(images)
            shape3=pred_img.shape

            #pred_ssm=softmax(pred_img)
            #_,pred_ssm=torch.max(pred_ssm,1)
            #pred_ssm.shape

            #loss = criterion(pred, mask)
            loss=getLoss(pred_img,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #Update the weights
            epoch_loss += loss.item()
 
            print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))

            # optimize weights

        torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

    # displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    with torch.no_grad():
        for _, (img, label) in enumerate(loader):
            shape = img.shape
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
            if gpu:
                img_torch = img_torch.cuda()
            pred = net(img_torch)
            pred_sm = softmax(pred)
            _,pred_label = torch.max(pred_sm,1)

            plt.subplot(1, 3, 1)
            plt.imshow(img*255.)
            plt.subplot(1, 3, 2)
            plt.imshow((label-1)*255.)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
            plt.show()

def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)

def softmax(input):
    # todo: implement softmax function
    p=input.exp() / (input.exp().sum())
    exp=torch.exp(input)
    p=exp/torch.sum(exp)
    return p

def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    p=choose(input,targets)
    m=targets.shape[0]
    #p=softmax(input)
    log_likelihood = -torch.log(p[range(m), targets])
    ce = torch.sum(log_likelihood) / m
    return ce

# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[x,y], x, y]
            i += 1

    pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])

    return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,


        data_dir=args.data_dir)
