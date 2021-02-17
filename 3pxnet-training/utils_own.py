import torch
import logging
import torchvision
import torchvision.transforms as transforms

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


import time
try:
   from .utils import *
except:
   from utils import *

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, verbal=False):
   losses = AverageMeter()
   top1 = AverageMeter()
   top5 = AverageMeter()

   for i, (inputs, target) in enumerate(data_loader):
      # measure data loading time.
      target = target.to(model.fc1.weight.device)
      input_var = Variable(inputs.to(model.fc1.weight.device))
      target_var = Variable(target)

      # compute output
      output = model(input_var)

      loss = criterion(output, target_var)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      losses.update(loss.item(), inputs.size(0))
      top1.update(prec1.item(), inputs.size(0))
      top5.update(prec5.item(), inputs.size(0))

      if training:
         # compute gradient and do SGD step
         optimizer.zero_grad()
         loss.backward()
         for p in model.modules():
            if hasattr(p, 'weight_org'):
               p.weight.data.copy_(p.weight_org)
         optimizer.step()
         for p in model.modules():
            if hasattr(p, 'weight_org'):
               p.weight_org.copy_(p.weight.data.clamp_(-1,1))
   if not training:
      if verbal:
         print('Epoch: [{0}]\t'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch, loss=losses, top1=top1, top5=top5))

   return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
   # switch to train mode
   model.train()
   return forward(data_loader, model, criterion, epoch, training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch, verbal=False):
   # switch to evaluate mode
   model.eval()
   return forward(data_loader, model, criterion, epoch, training=False, optimizer=None, verbal=verbal)

def perm_sort(weight):
   '''
   Form permutation list based on the RMS value of each column
   '''
   w_sum = torch.sum(torch.abs(weight), dim = 0)
   permute_list = np.argsort(w_sum.detach().data.cpu().numpy())
   permute_list = np.ascontiguousarray(np.flipud(permute_list))
   return torch.from_numpy(permute_list).type('torch.LongTensor').to(weight.device)

def permute_from_list(mask, permute_list, transpose=False):
   # (0. Reshape for 4D tensor). 1. Divide into sections. 2. Permute individually (3. Reshape back for 4D tensor)
   permute_redund = permute_list.size(0)
   if len(mask.size())==4:
      mask_flat = mask.permute(0,2,3,1).contiguous().view(-1, mask.size(1)).contiguous()
   else:
      mask_flat = mask.clone()
   split_size = int(np.ceil(float(mask_flat.size(0))/permute_redund).astype(int))
   mask_unpermute = torch.zeros_like(mask_flat)
   length = permute_list.size(1)
   mask_split = torch.split(mask_flat, split_size)
   permute_redund_cor = min(permute_redund, np.ceil(float(mask.size(0))/split_size).astype(int))
   for i in range(permute_redund_cor):
      if transpose:
         permute_list_t = torch.zeros_like(permute_list[i])
         permute_list_t[permute_list[i]] = torch.arange(length).to(permute_list.device)
         mask_unpermute[i*split_size:(i+1)*split_size] = mask_split[i][:,permute_list_t]
      else:
         mask_unpermute[i*split_size:(i+1)*split_size] = mask_split[i][:,permute_list[i]]

   if len(mask.size())==4:
      mask_unpermute = mask_unpermute.view(mask.size(0), mask.size(2), mask.size(3), mask.size(1)).permute(0,3,1,2)
   return mask_unpermute

def load_dataset(dataset):
   '''
   Give training/testing data loader and class information for several image datasets
   '''
   if dataset == 'MNIST':
      mean,std = (0.1307,), (0.3015,)
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
      trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
      testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
      classes = ('0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9')
   elif dataset == 'SVHN':
#       mean = [0.4377, 0.4438, 0.4728]
#       std = [0.1201, 0.1231, 0.1052]
#       transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
      transform = transforms.Compose([transforms.ToTensor()])
      trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                           download=True, transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                                        transforms.RandomHorizontalFlip(),
                                                                                        transforms.ToTensor()]))
#                                                                                         transforms.Normalize(mean,std)]))
      testset = torchvision.datasets.SVHN(root='./data', split='test',
                                          download=True, transform=transform)
      classes = ('0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9')
   elif dataset == 'CIFAR10':
      mean = [0.4914, 0.4822, 0.4465]
      std = [0.2023, 0.1994, 0.2010]
      transform = transforms.Compose([transforms.ToTensor()])
      trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transforms.Compose([transforms.RandomCrop(32, 4),
                                                                                           transforms.RandomHorizontalFlip(),
                                                                                           transforms.ToTensor(),
                                                                                           transforms.Normalize(mean,std)]))
      testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
      classes = ('plane', 'car', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   return trainset, testset, classes

def sim_n(tensor_group, tensor_new):
   '''
   Compute similarity score of a set of tensors and a new tensor
   '''
   height = tensor_group.size(0)
   width = tensor_group.size(1)+1
   tensor_group_new = torch.zeros((height, width)).to(tensor_group.device)
   tensor_group_new[:,:(width-1)] = tensor_group
   tensor_group_new[:,width-1] = tensor_new
   empty = torch.sum((tensor_group_new==0), dim=1).to(device=tensor_group.device, dtype=torch.float)
   compressed = torch.sum(tensor_group_new, dim=1)
   tensor_sum = -torch.sum(torch.min(empty, compressed))
   return tensor_sum

def sim_n_grad(grad_group, grad_new):
   raw_score = torch.sum(grad_group, dim=1)+grad_new
   score = -torch.abs(raw_score)
   tensor_sum = torch.sum(score)
   return tensor_sum

def one_time_permute(weight_gpu, thres, pack=8, weight_grad_gpu=None):
   '''
   Form the permutation list of a weight tensor given a sparsity constraint
   '''
   permute_list = list(range(weight_gpu.size(1)))
   permute_size = 0
   weight = (weight_gpu * weight_gpu)
   if type(weight_grad_gpu)!=type(None):
      weight_grad = weight_grad_gpu.cpu()
      grad_unprune = weight_grad * (weight==1).type(torch.FloatTensor) - weight_grad * (weight==-1).type(torch.FloatTensor) + abs(weight_grad) * (weight==0).type(torch.FloatTensor)
   # 1. Find the n-pack that has the maximum overlap
   counter = 0
   start_time = time.time()
   while permute_size+pack < weight.size(1):
      permute_list_valid = permute_list[permute_size:]

      result_tensor = torch.zeros((weight.size(0), pack)).to(weight_gpu.device)
      start_tensor = weight[:, permute_list_valid[0]]
      if type(weight_grad)!=type(None):
         start_gradient = grad_unprune[:, permute_list_valid[0]]
         result_gradient = torch.zeros((weight.size(0), pack))
         result_gradient[:,0] = start_gradient
      result_tensor[:,0] = start_tensor
      current_permute = [permute_list_valid[0]]
      permute_list_valid.remove(permute_list_valid[0])
      for i in range(1, pack, 1):
         max_score = -100000000
         max_index = -1
         for index in permute_list_valid:
            score_weight = sim_n(result_tensor[:,:i], weight[:,index])
            score = sim_n_grad(result_gradient[:,:i], grad_unprune[:,index])/100
            score = score_weight + score
            if score > max_score:
               max_score = score
               max_index = index
         result_tensor[:,i] = weight[:,max_index]
         result_gradient[:,i] = grad_unprune[:,max_index]
         permute_list_valid.remove(max_index)
         current_permute.append(max_index)

      # 2. Form permutation list such that these n columns are in left-most positions
      permute_list_finished = permute_list[:permute_size] + current_permute
      permute_list = permute_list_finished + [item for item in permute_list if item not in permute_list_finished]
      permute_size += pack
      counter += 1
   return torch.LongTensor(permute_list).to(weight_gpu.device)

def permute_all_weights_once(model, pack=8, mode=1):
   '''
   Determine permutation list of all modules of a network without pruning
   '''
   # Only permute. Pruning is done using something else
   import traceback
   for mod in model.modules():
      try:
         if isinstance(mod, nn.Linear):
            logging.info('Permuting '+ str(mod))
            cur_pack = pack
            permute_redund = mod.permute_list.size(0)
            section_size = np.ceil(float(mod.weight.size(0))/permute_redund).astype(int)
            permute_redund_cor = min(permute_redund, np.ceil(float(mod.weight.size(0))/section_size).astype(int))
            for i in range(permute_redund_cor):
               ceiling = min((i+1)*section_size, mod.weight.size(0))
               if mode==1:
                  mod.permute_list[i] = one_time_permute(mod.weight.data[i*section_size:ceiling], mod.thres, pack=cur_pack, weight_grad_gpu = mod.weight.grad[i*section_size:(i+1)*section_size])
               elif mode==0:
                  mod.permute_list[i] = perm_sort(mod.weight_org[i*section_size:ceiling])
               elif mode==2:
                  mod.permute_list[i] = perm_rand(mod.weight.data[i*section_size:ceiling])
         elif isinstance(mod, nn.Conv2d):
            logging.info('Permuting '+ str(mod))
            weight_flat = mod.weight.data.permute(0,2,3,1).contiguous().view(-1,mod.weight.size(1)).contiguous()
            grad_flat = mod.weight.grad.permute(0,2,3,1).contiguous().view(-1,mod.weight.size(1)).contiguous()
            if mode==1:
               mod.permute_list[0] = one_time_permute(weight_flat, mod.thres, pack=pack, weight_grad_gpu = grad_flat)
            elif mode==1:
               mod.permute_list[0] = perm_sort(weight_flat)
            elif mode==2:
               mod.permute_list[0] = perm_rand(weight_flat)
      except:
         traceback.print_exc()

def adjust_pack(net, pack):
   for mod in net.modules():
      if hasattr(mod, 'pack'):
         mod.pack -= mod.pack - pack
