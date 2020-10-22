import argparse
import os,stat
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

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

parser.add_argument('--save_dir', metavar='SAVE_DIR', default='./training_data/mnist/', type=str, help='Save dir')
parser.add_argument('--batch', metavar='BATCH', default=256, type=int, help='Batch size to use')
parser.add_argument('--size', metavar='SIZE', default=0, type=int, help='Size of network. 1 means large and 0 means small, 2 means CNN')
parser.add_argument('--full', metavar='FULL', default=0, type=int, help='If 1, train with full precision')
parser.add_argument('--binary', metavar='BINARY', default=0, type=int, help='If 1, train with dense binary')
parser.add_argument('--first_sparsity', metavar='FIRST_SPARSITY', default=0.9, type=float, help='Sparsity of the first layer')
parser.add_argument('--rest_sparsity', metavar='REST_SPARSITY', default=0.9, type=float, help='Sparsity of other layers')
parser.add_argument('--permute', metavar='PERMUTE', default=-1, type=int, help='Permutation method. -1 means no permutation. 0 uses sort, 1 uses group (method mentioned in paper)')
parser.add_argument('--seed', metavar='SEED', default=0, type=int, help='Seed to use for this run')
parser.add_argument('--cpu', dest='cpu', default=False, action='store_true', help='Train using cpu')
parser.add_argument('--test_start_id',default=0,help='start index of testing dataset')
parser.add_argument('--test_end_id',default=100,help='end index of testing dataset')



def main():
   global args, best_prec1
   args = parser.parse_args()
   seed = args.seed
   torch.manual_seed(seed)
   np.random.seed(seed)
   torch.backends.cudnn.deterministic=True

   dataset='MNIST'
   first_sparsity = args.first_sparsity
   rest_sparsity = args.rest_sparsity
   permute = args.permute
   size = args.size
   batch = args.batch
   pack = 32
   if args.full==1:
      full=True
   else:
      full=False
   if args.binary==1:
      binary=True
   else:
      binary=False

   device = 0

   test_start_id=int(args.test_start_id)
   test_end_id=int(args.test_end_id)
   if test_end_id<=test_start_id:
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
   if size==0:
      net = network.FC_small(full=full, binary=binary, first_sparsity=first_sparsity, rest_sparsity=rest_sparsity, align=True, ind=768, hid=128)
   elif size==1:
      net = network.FC_large(full=full, binary=binary, first_sparsity=first_sparsity, rest_sparsity=rest_sparsity, align=True, ind=768)
   elif size==2:
      net = network.CNN_tiny(full=full, binary=binary, conv_thres=first_sparsity, fc_thres=rest_sparsity, align=True)
   save_file = save_dir+'MNIST_s_{0}'.format(size)

   setup_logging(save_file + '_log.txt')
   logging.info("saving to %s", save_file)

   result_dic = save_file + '_result.pt'

   save_file += '.pt'

   if not args.cpu:
      torch.cuda.empty_cache()
      net.cuda(device)
   testdata=torch.tensor([2,2,2,2])
   testdata.cuda()


   learning_rate = 0.001
   criterion = nn.CrossEntropyLoss()
   lr_decay = np.power((2e-6/learning_rate), (1./100))

   optimizer = optim.Adam(net.parameters(), lr=learning_rate)
   scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

   ### Training section

   # Train without packing constraints
   utils_own.adjust_pack(net, 1)
   for epoch in range(0, 25):
      train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
      val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
      scheduler.step()

   # Retrain with permutation + packing constraint
   utils_own.adjust_pack(net, pack)
   utils_own.permute_all_weights_once(net, pack=pack, mode=permute)
   optimizer = optim.Adam(net.parameters(), lr=learning_rate)
   scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

   for epoch in range(0, 25):
      train_loss, train_prec1, train_prec5 = utils_own.train(trainloader, net, criterion, epoch, optimizer)
      val_loss, val_prec1, val_prec5 = utils_own.validate(testloader, net, criterion, epoch, verbal=True)
      scheduler.step()

   # Fix pruned packs and fine tune
   for mod in net.modules():
      if hasattr(mod, 'mask'):
         mod.mask = torch.abs(mod.weight.data)
   net.pruned = True
   optimizer = optim.Adam(net.parameters(), lr=learning_rate)
   scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)
   best_prec1 = 0

   for epoch in range(0, 200):
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
      scheduler.step()
   logging.info('\nTraining finished!')
   # Extract data
   conv_count = 0
   bn2d_count = 0
   bn1d_count = 0
   fc_count = 0

   creat_dir=True
   upload_dir = save_dir
   net.eval()
   net.finished=True
   for mod in net.modules():
      if isinstance(mod, nn.Conv2d):
         print(mod)
         if permute==1 and hasattr(mod,'permute_list'):
            lis = mod.permute_list.cpu().numpy()
            os.chdir('..')
            for root, dirs, files in os.walk("."):
               for name in dirs:
                  if fnmatch.fnmatch(name, "CNN_Tiny.nnef"):
                     creat_dir=False
            if creat_dir:
               os.mkdir("CNN_Tiny.nnef")
            os.chdir("CNN_Tiny.nnef")
            np.save('conv_{0}_list'.format(fc_count), lis)
            os.chdir('..')
            os.chdir('3pxnet-training')
         weight = mod.weight.data.type(torch.int16).cpu().numpy()
         conv_count += 1
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
         fc_count += 1

         if permute==1 and hasattr(mod,'permute_list'):
            lis = mod.permute_list.cpu().numpy()
            os.chdir('..')
            for root, dirs, files in os.walk("."):
               for name in dirs:
                  if size==0:
                     if fnmatch.fnmatch(name, "FC_Small.nnef"):
                        creat_dir=False
                  elif size==1:
                     if fnmatch.fnmatch(name, "FC_Large.nnef"):
                        creat_dir=False
                  elif size==2:
                     if fnmatch.fnmatch(name, "CNN_Tiny.nnef"):
                        creat_dir=False
            if creat_dir:
               if size == 0:
                  os.mkdir("FC_Small.nnef")
                  os.chdir("FC_Small.nnef")
               elif size == 1:
                  os.mkdir("FC_Large.nnef")
                  os.chdir("FC_Large.nnef")
               elif size == 2:
                  os.mkdir("CNN_Tiny.nnef")
                  os.chdir("CNN_Tiny.nnef")
            else:
               if size == 0:
                  os.chdir("FC_Small.nnef")
               elif size == 1:
                  os.chdir("FC_Large.nnef")
               elif size == 2:
                  os.chdir("CNN_Tiny.nnef")
            np.save('fc_{0}_list'.format(fc_count), lis)
            os.chdir('..')
            os.chdir('3pxnet-training')
         weight = mod.weight.data.type(torch.int16).cpu().numpy()
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
   fc_count=0
   if size ==0:
      x=Variable(torch.randn(1,784,requires_grad=True,device='cuda'))
      torch.onnx.export(net,x,"training_data/FC_Small.onnx",export_params=True,verbose=True,opset_version=9,input_names = ['input'], output_names = ['output'])
      model=onnx.load("training_data/FC_Small.onnx")
      # this can remove unecessary nodes
      ort_session = onnxruntime.InferenceSession("training_data/FC_Small.onnx")


   elif size==1:
      x=Variable(torch.randn(1,784,requires_grad=True,device='cuda'))
      torch.onnx.export(net,x,"training_data/FC_Large.onnx",export_params=True,verbose=True,input_names = ['input'], output_names = ['output'])
      model=onnx.load("training_data/FC_Large.onnx")
      # this can remove unecessary nodes
      ort_session = onnxruntime.InferenceSession("training_data/FC_Large.onnx")


   elif size==2:
      x=Variable(torch.randn(1,1,28,28,requires_grad=True,device='cuda'))
      torch.onnx.export(net,x,"training_data/CNN_Tiny.onnx",export_params=True,verbose=True,input_names = ['input'], output_names = ['output'])
      model=onnx.load("training_data/CNN_Tiny.onnx")
      # this can remove unecessary nodes
      ort_session = onnxruntime.InferenceSession("training_data/CNN_Tiny.onnx")
   device = torch.device('cpu')
   net.to(device)
   testdata=testloader.dataset.data[test_start_id:test_end_id]
   testdata=(testdata>0).float()+(-1*(testdata==0).float())
   if size==2:
      testdata=np.reshape(testdata,(test_end_id-test_start_id,1,28,28))
   print(testdata.shape)
   #label=testloader.dataset.targets[:100]
   #testdata.cuda()
   os.chdir('..')
   os.chdir('3pxnet-compiler')
   temp = open('__Golden.txt', 'w+')
   re=net(testdata)
   re=re.tolist()
   temp.write("Testing compiler output with golden output on MNIST \n")
   if permute==1:
      temp.write("The network is in 3PXNet style \n")
   else:
      if binary:
         temp.write("The network is dense and binary \n")
      else:
         temp.write("The network is pruned and packed \n")
   if size==0:
      temp.write("The network is a small FC network \n")
   elif size==1:
      temp.write("The network is a large FC network \n")
   else:
      temp.write("The network is a small CNN network \n")
   for i in range(test_end_id-test_start_id):
      temp.write(str(re[i].index(max(re[i]))))
      temp.write(' ')
   temp.close()
   '''
   for i,(data,label) in enumerate(testloader):
       if i==0:
           testdata=data
           testdata=testdata>0
           testdata=testdata.type('torch.FloatTensor')
           #testdata.cuda()
   print(net(testdata))
   '''
if __name__ == '__main__':
   main()
