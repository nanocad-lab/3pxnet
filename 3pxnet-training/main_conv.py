import argparse
import os
import torch
import logging

import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from utils import *
import utils_own
import network

import onnx
import onnxruntime
import torch.optim as optim
import fnmatch

parser = argparse.ArgumentParser(description='PyTorch CNN Training')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./training_data/conv/', help='Save dir')
parser.add_argument('--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('--size', metavar='SIZE', default=0, type=int, help='Size of network. 1 means large and 0 means medium')
parser.add_argument('--dataset', metavar='DATASET', default='SVHN', help='Dataset to train on. Currently choose from SVHN and CIFAR10')
parser.add_argument('--full', metavar='FULL', default=0, type=int, help='If 1, train with full precision')
parser.add_argument('--binary', metavar='BINARY', default=0, type=int, help='If 1, train with dense binary')
parser.add_argument('--conv_sparsity', metavar='CONV_SPARSITY', default=0.9, type=float, help='Sparsity of the convolutional layers')
parser.add_argument('--fc_sparsity', metavar='FC_SPARSITY', default=0.9, type=float, help='Sparsity of fully-connected layers')
parser.add_argument('--permute', metavar='PERMUTE', default=-1, type=int, help='Permutation method. -1 means no permutation')
parser.add_argument('--pad', metavar='PAD', default=0, type=int, help='Padding value used')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='Seed to use for this run')
parser.add_argument('--cpu', dest='cpu', default=False, action='store_true', help='Train using cpu')
parser.add_argument('--test_start_id',default=0,help='start index of testing dataset')
parser.add_argument('--test_end_id',default=100,help='end index of testing dataset')
parser.add_argument('--device',default=0,type=int,help='GPU to use')


def main():
    global args, best_prec1
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    dataset = args.dataset
    conv_sparsity = args.conv_sparsity
    fc_sparsity = args.fc_sparsity
    permute = args.permute
    size = args.size
    batch = args.batch
    pad = args.pad
    pack = 32
    if args.full==1:
        full=True
    else:
        full=False
    if args.binary==1:
        binary=True
    else:
        binary=False
    device = args.device

    test_start_id = int(args.test_start_id)
    test_end_id = int(args.test_end_id)
    if test_end_id <= test_start_id:
        print("ERROR: test end id is less than test start id")
        exit(1)

    trainset, testset, classes = utils_own.load_dataset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                             shuffle=False, num_workers=2)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.size==1:    
        net = network.CNN_large(full=full, binary=binary, conv_thres=conv_sparsity, fc_thres=fc_sparsity, align=True, pad=pad)
    elif args.size==0:
        net = network.CNN_medium(full=full, binary=binary, conv_thres=conv_sparsity, fc_thres=fc_sparsity, align=True, pad=pad)
        
    save_file = save_dir + '{0}_s_{1}'.format(dataset, size)

    setup_logging(save_file + '_log.txt')
    logging.info("saving to %s", save_file)
    
    save_file += '.pt'
    regime = getattr(net, 'regime', {0: {'optimizer': 'Adam', 'lr': 0.1,'momentum': 0.9,'weight_decay': 1e-4}})  
    
    if not args.cpu:
        torch.cuda.empty_cache()
        net.cuda(device)
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    lr_decay = np.power((2e-6/learning_rate), (1./100))

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    ### Training section
    
    # Train without packing constraints
    utils_own.adjust_pack(net,1)
    for epoch in range(0, 25):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
    
    # Retrain with permutation + packing constraint
    utils_own.adjust_pack(net, pack)
    utils_own.permute_all_weights_once(net, pack=pack, mode=permute)
    
    for epoch in range(0, 25):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
    
    # Fix pruned packs and fine tune
    for mod in net.modules():
        if hasattr(mod, 'mask'):
            mod.mask = torch.abs(mod.weight.data)    
    net.pruned = True
    best_prec1 = 0
    
    for epoch in range(0, 200):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
        val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=False)
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        

        if is_best:
            torch.save(net, save_file)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec {train_prec1:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec {val_prec1:.3f} \t'
                         .format(epoch+1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1))
    logging.info('\nTraining finished')
    
    # Extract data
    conv_count = 0
    bn2d_count = 0
    bn1d_count = 0
    fc_count = 0

    upload_dir = save_dir
    creat_dir=True
    for mod in net.modules():
        if isinstance(mod, nn.Conv2d):
            print(mod)
            weight = mod.weight.data.type(torch.int16).cpu().numpy()
            
            conv_count += 1
            if permute==1 and hasattr(mod,'permute_list'): 
                lis = mod.permute_list.cpu().numpy()
                os.chdir('..')
                for root, dirs, files in os.walk("."):
                    for name in dirs:
                        if size == 0:
                            if fnmatch.fnmatch(name, "CNN_Medium.nnef"):
                                creat_dir = False
                        else:
                            if fnmatch.fnmatch(name, "CNN_Large.nnef"):
                                creat_dir = False
                if creat_dir:
                    if size == 0:
                        os.mkdir("CNN_Medium.nnef")
                        os.chdir("CNN_Medium.nnef")
                    else:
                        os.mkdir("CNN_Large.nnef")
                        os.chdir("CNN_Large.nnef")
                else:
                    if size == 0:
                        os.chdir("CNN_Medium.nnef")
                    else:
                        os.chdir("CNN_Large.nnef")
                np.save('conv_{0}_list'.format(conv_count), lis)   
                os.chdir('..')
                os.chdir('3pxnet-training')
            np.save(upload_dir+'conv_{0}_weight'.format(conv_count), weight)
        if isinstance(mod, nn.BatchNorm2d):
            print(mod)
            weight = mod.weight.data.cpu().numpy()
            bias = mod.bias.data.cpu().numpy()
            mean = mod.running_mean.cpu().numpy()
            var = mod.running_var.cpu().numpy()

            bn2d_count += 1
            np.save(upload_dir+'bn2d_{0}_weight'.format(bn2d_count), weight)
            np.save(upload_dir+'bn2d_{0}_bias'.format(bn2d_count), bias)
            np.save(upload_dir+'bn2d_{0}_mean'.format(bn2d_count), mean)
            np.save(upload_dir+'bn2d_{0}_var'.format(bn2d_count), var)

        if isinstance(mod, nn.Linear):
            print(mod)
            weight = mod.weight.data.type(torch.int16).cpu().numpy()
            fc_count += 1
            if permute==1 and hasattr(mod,'permute_list'): 
                lis = mod.permute_list.cpu().numpy()
                os.chdir('..')
                for root, dirs, files in os.walk("."):
                    for name in dirs:
                        if size==0:
                            if fnmatch.fnmatch(name, "CNN_Medium.nnef"):
                                creat_dir=False
                        else:
                            if fnmatch.fnmatch(name, "CNN_Large.nnef"):
                                creat_dir=False
                if creat_dir:
                    if size==0:
                        os.mkdir("CNN_Medium.nnef")
                        os.chdir("CNN_Medium.nnef")
                    else:
                        os.mkdir("CNN_Large.nnef")
                        os.chdir("CNN_Large.nnef")
                else:
                    if size==0:
                        os.chdir("CNN_Medium.nnef")
                    else:
                        os.chdir("CNN_Large.nnef")
                np.save('fc_{0}_list'.format(fc_count), lis)
                os.chdir('..')
                os.chdir('3pxnet-training')
            np.save(upload_dir+'fc_{0}_weight'.format(fc_count), weight)
        if isinstance(mod, nn.BatchNorm1d):
            print(mod)
            bn1d_count += 1
            if type(mod.weight) != type(None):
                weight = mod.weight.data.cpu().numpy()
                np.save(upload_dir+'bn1d_{0}_weight'.format(bn1d_count), weight)
            if type(mod.bias) != type(None):
                bias = mod.bias.data.cpu().numpy()
                np.save(upload_dir+'bn1d_{0}_bias'.format(bn1d_count), bias)
            mean = mod.running_mean.cpu().numpy()
            var = mod.running_var.cpu().numpy()

            np.save(upload_dir+'bn1d_{0}_mean'.format(bn1d_count), mean)
            np.save(upload_dir+'bn1d_{0}_var'.format(bn1d_count), var)

    if args.size==1:
        if args.cpu:
            x=Variable(torch.randn(1,3,32,32,requires_grad=True))
        else:
            x=Variable(torch.randn(1,3,32,32,requires_grad=True).cuda(device))
        torch.onnx.export(net,x,"training_data/CNN_Large.onnx",verbose=True,opset_version=9,input_names = ['input'], output_names = ['output'])
        model=onnx.load("training_data/CNN_Large.onnx")
        # this can remove unecessary nodes
        ort_session = onnxruntime.InferenceSession("training_data/CNN_Large.onnx")
    elif args.size==0:
        if args.cpu:
            x=Variable(torch.randn(1,3,32,32,requires_grad=True))
        else:
            x=Variable(torch.randn(1,3,32,32,requires_grad=True).cuda(device))
        torch.onnx.export(net,x,"training_data/CNN_Medium.onnx",verbose=True,opset_version=9,input_names = ['input'], output_names = ['output'])
        model=onnx.load("training_data/CNN_Medium.onnx")
        # this can remove unecessary nodes
        ort_session = onnxruntime.InferenceSession("training_data/CNN_Medium.onnx")
    device = torch.device('cpu')
    net.to(device)
    testdata=torch.from_numpy(testloader.dataset.data[test_start_id:test_end_id]).permute(0,3,1,2).float()
    os.chdir('..')
    os.chdir('3pxnet-compiler')
    temp = open('__Golden.txt', 'w+')
    re = net(testdata)
    re = re.tolist()
    temp.write("Testing compiler output with golden output on CIFAR \n")
    if permute == 1:
        temp.write("The network is in 3PXNet style \n")
    else:
        if binary:
            temp.write("The network is dense and binary \n")
        else:
            temp.write("The network is pruned and packed \n")
    if size == 0:
        temp.write("The network is a small CNN network \n")
    else:
        temp.write("The network is a large CNN network \n")
    for i in range(test_end_id-test_start_id):
        temp.write(str(re[i].index(max(re[i]))))
        temp.write(' ')
    temp.close()

if __name__ == '__main__':
    main()
