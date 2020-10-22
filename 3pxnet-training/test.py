import argparse
import torch
import os
import numpy as np
import onnxruntime as ort
def to_numpy(tensor):
   return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
parser = argparse.ArgumentParser(description='automatically generate inference code')
parser.add_argument('--input', help="""name of input directory. This should be the converted ONNX "
      "formatted neural network""")
parser.add_argument('--dataset', metavar='DATASET', default='MNIST',
                       help='Dataset trained on. Currently choose from MNIST and CIFAR10')
args = parser.parse_args()
in_model=args.input
dataset=args.dataset
from utils_own import load_dataset
trainset, testset, classes = load_dataset(dataset)
testloader=torch.utils.data.DataLoader(testset, batch_size=256,
                                            shuffle=False, num_workers=2)
testdata=testloader.dataset.data[:100]
if dataset=='MNIST':
   testdata=(testdata>0).float()+(-1*(testdata==0).float())
ort_sess = ort.InferenceSession(in_model)
os.chdir('..')
os.chdir('3pxnet-compiler')
temp = open('__Golden.txt', 'w+')
re=[]
if dataset=='MNIST':
   temp.write("Testing compiler output with golden output on MNIST \n")
else:
   temp.write("Testing compiler output with golden output on CIFAR \n")
for i in range(0,100):
   if dataset=='MNIST':
      reshaped=testdata[i].view(1,-1)
      ort_input={ort_sess.get_inputs()[0].name: reshaped.numpy()}
   else:
      ort_input = {ort_sess.get_inputs()[0].name: testdata[i].numpy()}
   re.append(ort_sess.run(None, ort_input)[0].tolist()[0])
for i in range(100):
   temp.write(str(re[i].index(max(re[i]))))
   temp.write(' ')
temp.close()
