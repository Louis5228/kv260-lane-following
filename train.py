import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader

import argparse
import sys
import os
import shutil

from common import *


DIVIDER = '-----------------------------------------'


def train_test(build_dir, batchsize, learnrate, epochs):

    float_model = build_dir + '/float_model'

    # detect if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    # instantiate network
    model = CNN_Model().to(device)

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # define an optimizer
    optimizer = optim.SGD(model.parameters(), lr=learnrate, momentum=0.9)

    # prepare dataset
    train_data = datasets.ImageFolder(
        'dataset/Trail_dataset/train_data',
        transform = transforms.Compose([transforms.ToTensor()])                         
    )

    test_data = datasets.ImageFolder(
        'dataset/Trail_dataset/test_data',
        transform = transforms.Compose([transforms.ToTensor()])                         
    )

    # data loaders
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=batchsize, 
                              shuffle= True,
                              num_workers=2)

    test_loader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             shuffle=True,
                             num_workers=2)

    # training
    train(model, device, train_loader, optimizer, criterion, epochs)

    # testing
    test(model, device, test_loader)

    # save the trained model
    shutil.rmtree(float_model, ignore_errors=True)    
    os.makedirs(float_model)   
    save_path = os.path.join(float_model, 'f_model.pth')
    torch.save(model.state_dict(), save_path)
    print('Trained model written to', save_path)

    return

def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='build',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=16,            help='Training batchsize. Must be an integer. Default is 16')
    ap.add_argument('-e', '--epochs',      type=int,  default=5,             help='Number of training epochs. Must be an integer. Default is 5')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print(DIVIDER)

    train_test(args.build_dir, args.batchsize, args.learnrate, args.epochs)

    return


if __name__ == '__main__':
    run_main()
