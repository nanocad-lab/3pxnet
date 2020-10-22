import nnef
import os, fnmatch
import argparse
import struct
from bitarray import bitarray
import numpy as np
import string
'''
Compiler for 3PXNet. Compiles a neural network stored in NNEF format to C using inference engine.
'''
'''
NOTIFICATIONS:
   variablen is a dictionary, its keys are variable_# and value is a list, [non-pruned inputs, the operation object whose output name is variable_#]
   variables is also a dictionary, its keys are variable_# and value is the file name of this data
   batchn is also a dictionary, its keys are indices of graph object, where there is a batchnorm operation, the values are variable_#, who are input parameter to this operation 
   The workflow is like this: given a graph, first this program will read all of its operations and determine whether a given operation is able to be compiled or not.
   Then it reads the data files and put the values into header files, i.e., decoding_data. After that, threshold and sign needed for some batchnorm layers are computed. 
   Then it starts writing source file. total_ops stores indices of graph where matrix multiplication, whether conv or fc, takes place.
   To decide whether there is a batchnorm to one layer or not, it look ahead for another matrix multiplication, if there is a batchnorm operation between these two, then there will be a 
   batchnorm, and vice versa.
'''

class convert(object):
   def __init__(self,input_dir,dataset,test_start_id,test_end_id):
      '''
      initialize a convert object

      :param input_dir: the input directory, its name should end with .nnef
      :param dataset: dataset to test against
      :param test_start_id: testing dataset start index
      :param test_end_id: testing dataset end index
      '''
      self.input_dir=input_dir
      self.dataset=dataset
      self.test_start_id=int(test_start_id)
      self.test_end_id=int(test_end_id)
      self.graph=nnef.Graph
      # batch norm variables
      self.var = {}
      self.mean = {}
      self.gamma = {}
      self.beta = {}
      # store variables with their names as keys
      # for specific information, please see NOTIFICATIONS above
      self.variablen = {}
      self.variables = {}
      # store index of propagation in the graph
      self.matrixmul = []
      self.conv = []
      self.batchn = {}
      #input shape
      self.in_shape = []
      self.rank = []
      # which batch norm layer is the last one
      self.batch_last=" "
      # source code we are writing to
      self.source=0
      #permutation list. For specific information, please see training engine
      #as well as the paper the whole project is based on.
      self.list=[]
      self.tempweight=[]
      self.tempsparse=False
      self.tempoutput=0
      self.name=" "
      self.lastlist=[]
      self.tempvar=" "
      self.tempmean=""
      self.tempgamma=""
      self.tempbeta=""

   def replace_non_ascii(self,stri):
      '''
         Replace all non ascii, including . , in the file name to _

      :param stri: input string
      :return: the input string with non-character or non-digit being replaced by _
      '''
      return ''.join([i if i in string.ascii_letters or i in string.digits else '_' for i in stri])


   def search_non_ascii(self,stri):
      '''
         Search for the first letter that is not letter or digit
         Needed for determine last layer of batch norm

      :param stri: input string
      :return: the first index of non-character or non-digit char
      '''
      for i in range(len(stri)):
         if not (stri[i] in string.ascii_letters or stri[i] in string.digits):
            return i





   def loadgraph(self):
      '''
      load the nnef graph into compiler
      '''
      print(self.input_dir)
      os.chdir(self.input_dir)
      if "graph.nnef" not in os.listdir("."):
         print("ERROR: BAD NNEF DIRECTORY!")
         exit()
      else:
         self.graph = nnef.load_graph('graph.nnef')
      print("Graph loaded")

   def find_batch_last(self):
      '''
         Determines the last layer
         The last layer will not be binarized, so batchnorm has to be delt with differently
         Requires NNEF graph to be loaded (using loadgraph())

      :return: NA
      '''
      #find out the last matrix multiplication or convolution operation
      batch_last = next(i for i in reversed(range(len(self.graph.operations))) if
                        self.graph.operations[i].name == 'matmul' or self.graph.operations[i].name == 'conv')
      # If True, last layer is batch norm, otherwise false
      lastBnFound=False
      #if there is batch norm after the last matmul or conv, then that is the last batch norm layer
      for i in range(batch_last,len(self.graph.operations)):
         if self.graph.operations[i].name=='batch_normalization':
            for ops in self.graph.operations:
               # Get the variable which is the input to the batch_normalization layer
               if ops.outputs['output']==self.graph.operations[i].inputs['mean']: #get the name for the last batcch norm layer
                  batch_last=ops.attribs['label']
                  lastBnFound=True
                  break
      if lastBnFound:
         # If found, save the label for the last batchnorm layer
         self.batch_last=batch_last[0:self.search_non_ascii(batch_last)] #cutoff the ".mean" part from batch_last
      else:
         self.batch_last=" "

   def write_source_first(self):
      '''
         Write to the source file include headers for variables

      :return: NA
      '''
      os.chdir("../3pxnet-compiler")
      if "autogen" not in os.listdir("."):
         os.mkdir("autogen")
      os.chdir("autogen")
      source = open("source.c", 'w+')
      self.source=source
      # Write generic headers
      source.write("#include <stdio.h>\n")
      source.write("#include <stdlib.h>\n")
      source.write("#include <stdint.h>\n")
      source.write("#include <string.h>\n")
      source.write("#include <math.h>\n")
      source.write("#include <time.h>\n")
      source.write("#include <errno.h>\n")
      # Write model specific headers
      # TODO: don't include headers that are not neeed for a particular model
      source.write("#include \"datatypes.h\"\n")
      source.write("#include \"utils.h\"\n")
      source.write("#include \"xnor_base.h\"\n")
      source.write("#include \"xnor_fc.h\"\n")
      source.write("#include \"3pxnet_fc.h\"\n")
      source.write("#include \"3pxnet_cn.h\"\n")
      source.write("#include \"xnor_fc.h\"\n")
      source.write("#include \"bwn_dense_cn.h\"\n")
      os.chdir("..")
      os.chdir(self.input_dir)

   def writefc(self, write, rank, temp_array, sparse, output, name):
      '''
      Write fc layer's data into C headers

      :param write: whether to write or not(related to permutation issue)
      :param rank: weight's shape
      :param temp_array: weight data
      :param sparse: whether the layer is sparse or not
      :param output: IO object, corresponding to the header file it's writing to
      :param name: name of the header file
      :return: indices: if it is a sparse layer, indices are used to calculate # non-pruned inputs
      '''
      indices = []
      if write:
         print("Writing to header " + name + ".h ...")
         output.write("#define _" + name + " {\\\n")
      # NNEF format weight values are stored in row-major order.
      # So for a fc layer, its shape is [input, output]
      for i in range(rank[1]):
         # outtemp is used to store packs
         # mask is used to check whether a given pack is all zero
         outtemp = bitarray()
         mask = bitarray()
         for j in range(rank[0]):
            temp = temp_array[j, i]
            if temp >= 0:
               outtemp.append(1)
            else:
               outtemp.append(0)
            mask.append(temp == 0)
            if j % 32 == 31:
               if sparse:
                  # a pack is all zero
                  if int(mask.to01(), 2) == 2 ** 32 - 1:
                     outtemp = bitarray()
                     mask = bitarray()
                  else:
                     if write:
                        output.write(str("0x%x" % int(outtemp.to01(), 2)) + ", \\\n")
                     indices.append(int(j % rank[0] / 32))
                     outtemp = bitarray()
                     mask = bitarray()
               else:
                  if write:
                     output.write(str("0x%x" % int(outtemp.to01(), 2)) + ", \\\n")
                  outtemp = bitarray()
                  mask = bitarray()
      if write:
         output.write("}\n")
         if sparse:
            output.write("#define _" + name + "_indices {\\\n")
            for i in range(len(indices)):
               output.write(str(indices[i]) + ", \\\n")
            output.write("}\n")
         output.close()
      return indices

   def writecn(self, write, rank, temp_array, sparse, output, name):
      '''
      Write conv layer's data into C headers
      The same as fc layer, NNEF format stores value in row-major order
      So for a conv layer, the shape is [n,z,y,x]
      But, I modified this order during decoding data time.
      So now the input rank has a shape [x,y,z,n]

      :param write: whether to write or not(related to permutation issue)
      :param rank: weight's shape
      :param temp_array: weight data
      :param sparse: whether the layer is sparse or not
      :param output: IO object, corresponding to the header file it's writing to
      :param name: name of the header file
      :return: indices: if it is a sparse layer, indices are used to calculate # non-pruned inputs
      '''
      indices = []
      if write:
         print("Writing to header " + name + '.h ...')
         output.write("#define _" + name + " {\\\n")
      for n in range(rank[0]):
         # outtemp is used to store packs
         # mask is used to check whether a given pack is all zero
         outtemp = bitarray()
         mask = bitarray()
         for y in range(rank[2]):
            for x in range(rank[3]):
               for z in range(rank[1]):
                  temp = temp_array[x, y, z, n]
                  if temp >= 0:
                     outtemp.append(1)
                  else:
                     outtemp.append(0)
                  mask.append(temp == 0)
                  if z % 32 == 31:
                     if sparse:
                        # a pack is all zero
                        if int(mask.to01(), 2) == 2 ** 32 - 1:
                           outtemp = bitarray()
                           mask = bitarray()
                        else:
                           if write:
                              output.write(str("0x%x" % int(outtemp.to01(), 2)) + ", \\\n")
                           indices.append(int(z / 32) + x * int(rank[1] / 32) + y * rank[3] * int(
                              rank[1] / 32))
                           outtemp = bitarray()
                           mask = bitarray()
                     else:
                        if write:
                           output.write(str("0x%x" % int(outtemp.to01(), 2)) + ", \\\n")
                        outtemp = bitarray()
                        mask = bitarray()
      if write:
         output.write("}\n")
         if sparse:
            output.write("#define _" + name + "_indices {\\\n")
            for i in range(len(indices)):
               output.write(str(indices[i]) + ", \\\n")
            output.write("}\n")
         output.close()
      return indices

   def decoding_data(self, input, output, name, last, identity, first):
      '''
      Processing a given .dat file stored in NNEF format
      To be specific, a NNEF formatted neural network contains a .graph file and several .dat files.
      This function deals with a given .dat file. it first reads in specifications of this file, such as
      Its length and its shape. Then it will translate weights stored in binary in this .dat file into packs or digits.
      The actual writing-to-header process is done by writecn and writefc functions.

      :param input: IO object, corresponding to the .dat file it's reading from
      :param output: IO object, corresponding to the header file it's writing to
      :param name: name of the header file
      :param last: whether the given .dat file is corresponded with the last batch norm layer
      :param identity: whether the given .dat file contains values for conv/fc/batchnorm layer
      :param first: whether it is the first matrix operations in the graph. Needed for permutation issue
      :return: if the input file is a fc or cn layer and it's sparse, then non-pruned inputs number is returned.
            if the input file is a batch norm layer and it's not the last one, a list with all its values are returned.
            otherwise, return 0
      '''
      # Skip NNEF header
      input.read(4)
      # Length of the data in bytes
      length = int.from_bytes(input.read(4), byteorder='little')
      # Number of dimensions the data
      rank_n = int.from_bytes(input.read(4), byteorder='little')
      rank = []  # n,z,y,x
      # Determine layer type
      batch = (identity == 0)
      fc = (identity == 1)
      cn = (identity == 2)
      # Get dimension sizes
      for i in range(0, rank_n):
         rank.append(int.from_bytes(input.read(4), byteorder='little'))
      # Skip padding
      input.read((8 - rank_n) * 4)
      bits_per_item = int.from_bytes(input.read(4), byteorder='little')
      input.read(2)
      size = int(bits_per_item / 8)
      # interpret as float or int
      # Variables used for quantization
      algo = int.from_bytes(input.read(2), byteorder='big')
      signess = int.from_bytes(input.read(4), byteorder='little')
      # TODO: more about linear and log quantize later
      # reference: https://www.khronos.org/registry/NNEF/specs/1.0/nnef-1.0.2.html#container-structure
      input.seek(128, 0)
      # start reading data
      # Flag for sparse operations
      sparse = False
      indices = []
      result = []
      # fc needs to be packed in column-major order
      if fc:
         # Holds decoded weight values
         temp_array = np.zeros((rank[0], rank[1]))
         for i in range(rank[0]):
            for j in range(rank[1]):
               temp = list(input.read(size))
               # changing endianess
               for b in range(0, int(len(temp) / 2)):
                  temp1 = temp[b]
                  temp[b] = temp[len(temp) - b - 1]
                  temp[len(temp) - b - 1] = temp1
               temp = bytes(temp)
               # decode as float
               # If there is a zero, treat as sparse
               if struct.unpack('!f', temp)[0] == 0:
                  sparse = True
               temp_array[i, j] = struct.unpack('!f', temp)[0]
         # permutation
         os.chdir('..')
         os.chdir(self.input_dir)
         # True if permutation is required
         flag = False
         for root, dirs, files in os.walk("."):
            for name1 in files:
               if fnmatch.fnmatch(name1.replace('_', ''), name.replace('weight', 'list.npy').replace('_', '')):
                  print("Permuting...")
                  flag = True
                  temp_weight = np.zeros((rank[0], rank[1]))
                  permute_list = np.load(name1)
                  if first:
                     self.list = permute_list
                  # permute input channel for current layer so that we can pack weights
                  for i in range(rank[0]):
                     temp_weight[i, 0:] = np.copy(temp_array[permute_list[0, i], 0:])
                  # permute output channel for last layer so that channels match
                  if len(self.tempweight) != 0:
                     tt = np.copy(self.tempweight)
                     if len(tt.shape) == 4:
                        for j in range(tt.shape[3]):
                           self.tempweight[0:, 0:, 0:, j] = np.copy(tt[0:, 0:, 0:, permute_list[0, j]])
                        self.writecn(True, [tt.shape[3], tt.shape[2], tt.shape[1], tt.shape[0]], self.tempweight,
                                     self.tempsparse, self.tempoutput, self.name)
                     else:
                        for i in range(rank[0]):
                           self.tempweight[0:, i] = np.copy(tt[0:, permute_list[0, i]])
                        self.writefc(True, tt.shape, self.tempweight, self.tempsparse, self.tempoutput, self.name)
                     # permute the last batch layer as well
                     tt = np.copy(self.var[self.tempvar])
                     for i in range(rank[0]):
                        self.var[self.tempvar][i] = np.copy(tt[permute_list[0, i]])
                     tt = np.copy(self.mean[self.tempmean])
                     for i in range(rank[0]):
                        self.mean[self.tempmean][i] = np.copy(tt[permute_list[0, i]])
                     tt = np.copy(self.gamma[self.tempgamma])
                     for i in range(rank[0]):
                        self.gamma[self.tempgamma][i] = np.copy(tt[permute_list[0, i]])
                     tt = np.copy(self.beta[self.tempbeta])
                     for i in range(rank[0]):
                        self.beta[self.tempbeta][i] = np.copy(tt[permute_list[0, i]])
                  temp_array = temp_weight
                  # save this layer's state so that later we can permute its output channel
                  self.tempweight = temp_array
                  self.tempoutput = output
                  self.tempsparse = sparse
                  self.name = name
                  self.lastlist = permute_list
                  break
         # if there is nothing to be permuted, meaning this layer is not on the temp state in this class
         # so we directly write them into header file
         # otherwise, wait for it to be permuted by next layer
         if flag:
            indices = self.writefc(False, rank, temp_array, sparse, output, name)
         else:
            indices = self.writefc(True, rank, temp_array, sparse, output, name)
         os.chdir("../3pxnet-compiler/autogen")

      elif cn:
         # first layer in a cnn
         # it uses binarized dense layer, so we don't pack it
         if rank[1] % 32 != 0:
            output.write("#define _" + name + " {\\\n")
            temp_array = np.zeros((rank[0], rank[1], rank[2], rank[3]))
            for n in range(rank[0]):
               for z in range(rank[1]):
                  for y in range(rank[2]):
                     for x in range(rank[3]):
                        temp = list(input.read(size))
                        # changing endianess
                        for b in range(0, int(len(temp) / 2)):
                           temp1 = temp[b]
                           temp[b] = temp[len(temp) - b - 1]
                           temp[len(temp) - b - 1] = temp1
                        temp = bytes(temp)
                        if struct.unpack('!f', temp)[0] == 0:
                           sparse = True
                        temp_array[n, z, y, x] = struct.unpack('!f', temp)[0]
            print("Sparse?: " + str(sparse))
            for n in range(rank[0]):
               for y in range(rank[2]):
                  for x in range(rank[3]):
                     for z in range(rank[1]):
                        temp = temp_array[n, z, y, x]
                        output.write(str(int(temp)) + ", ")
                  output.write('\\\n')
            output.write("}\n")
            output.close()
         # other conv layers in a cnn
         else:
            temp_array = np.zeros((rank[3], rank[2], rank[1], rank[0]))
            for n in range(rank[0]):
               for z in range(rank[1]):
                  for y in range(rank[2]):
                     for x in range(rank[3]):
                        temp = list(input.read(size))
                        # changing endianess
                        for b in range(0, int(len(temp) / 2)):
                           temp1 = temp[b]
                           temp[b] = temp[len(temp) - b - 1]
                           temp[len(temp) - b - 1] = temp1
                        temp = bytes(temp)
                        if struct.unpack('!f', temp)[0] == 0:
                           sparse = True
                        temp_array[x, y, z, n] = struct.unpack('!f', temp)[0]
            print("Sparse?: " + str(sparse))
            # permutation
            os.chdir('..')
            os.chdir(self.input_dir)
            flag = False
            for root, dirs, files in os.walk("."):
               for name1 in files:
                  if fnmatch.fnmatch(name1.replace('_', ''), name.replace('weight', 'list.npy').replace('_', '')):
                     print("Permuting...")
                     flag = True
                     temp_weight = np.zeros((rank[3], rank[2], rank[1], rank[0]))
                     permute_list = np.load(name1)
                     if first:
                        self.list = permute_list
                     # permute input channel of current layer
                     for j in range(rank[0]):
                        for i in range(rank[1]):
                           temp_weight[0:, 0:, i, j] = np.copy(temp_array[0:, 0:, permute_list[0, i], j])
                     # permute output channel of last layer
                     # since it's not possible to have a fc layer before a conv layer,
                     # we don't consider that case here.
                     if len(self.tempweight) != 0:
                        tt = np.copy(self.tempweight)
                        for j in range(tt.shape[3]):
                           self.tempweight[0:, 0:, 0:, j] = np.copy(tt[0:, 0:, 0:, permute_list[0, j]])
                        self.writecn(True, [tt.shape[3], tt.shape[2], tt.shape[1], tt.shape[0]],
                                     self.tempweight, self.tempsparse, self.tempoutput, self.name)
                        # permute the last batch layer as well
                        tt = np.copy(self.var[self.tempvar])
                        for i in range(rank[0]):
                           self.var[self.tempvar][i] = np.copy(tt[permute_list[0, i]])
                        tt = np.copy(self.mean[self.tempmean])
                        for i in range(rank[0]):
                           self.mean[self.tempmean][i] = np.copy(tt[permute_list[0, i]])
                        tt = np.copy(self.gamma[self.tempgamma])
                        for i in range(rank[0]):
                           self.gamma[self.tempgamma][i] = np.copy(tt[permute_list[0, i]])
                        tt = np.copy(self.beta[self.tempbeta])
                        for i in range(rank[0]):
                           self.beta[self.tempbeta][i] = np.copy(tt[permute_list[0, i]])
                     temp_array = temp_weight
                     # save this layer's state so that later we can permute its output channel
                     self.tempweight = temp_array
                     self.tempoutput = output
                     self.tempsparse = sparse
                     self.name = name
                     self.lastlist = permute_list
                     break
            if flag:
               indices = self.writecn(False, rank, temp_array, sparse, output, name)
            else:
               indices = self.writecn(True, rank, temp_array, sparse, output, name)
            os.chdir("../3pxnet-compiler/autogen")

      # batchnorm
      else:
         if last:
            print("Writing to header " + name + ".h ...")
         for i in range(int(length / size)):
            # One great feature of NNEF is it doesn't use many concrete data types. Therefore, there are several
            # encoding algorithms provided. Since current training engine will not train weights whose data types
            # are not float, this converter does not support any other encoding algorithm
            # TODO: depending on encoding algorithm, theoretically we should decode numbers in different ways
            # TODO: more support for this later
            # reference: https://www.khronos.org/registry/NNEF/specs/1.0/nnef-1.0.2.html#container-structure
            if algo == 0:
               temp = list(input.read(size))
               # changing endianess
               for j in range(0, int(len(temp) / 2)):
                  temp1 = temp[j]
                  temp[j] = temp[len(temp) - j - 1]
                  temp[len(temp) - j - 1] = temp1
               temp = bytes(temp)
               if last and "var" in name:
                  # what we really need is standard deviation, not variance
                  # because it will be considered in inference
                  output.write(str(np.sqrt(struct.unpack('!f', temp)[0])) + ", \\\n")
               elif last:
                  output.write(str(struct.unpack('!f', temp)[0]) + ", \\\n")
               else:
                  result.append(struct.unpack('!f', temp)[0])
            elif algo == 1 and signess == 0:
               output.write(str(int.from_bytes(input.read(size), byteorder='little')) + ", \\\n")
            else:
               output.write(str(int.from_bytes(input.read(size), byteorder='little', signed=True)) + ", \\\n")
      if last:
         output.write("}\n")
      if batch:
         return result
      # return non-pruned input number
      elif sparse and fc:
         return int(32 * len(indices) / rank[1])
      elif sparse and cn:
         return int(32 * len(indices) / rank[0])
      else:
         return 0

   def processing_graph(self):
      '''
         For every operation in the graph, determine whether we can translate into C using inference engine or not
         If not, then there will be WARNING or ERROR printed on screen
         If we can, then corresponding data files are decoded and written
         It uses a lot of dictionary type data structures. For detailed information, please see NOTIFICATIONS

      :return: NA
      '''
      rank=[]
      i=0
      # NNEF loaded graph.operations is guaranteed to be in order
      for ops in self.graph.operations:
         print("-----------------------------------------")
         print("Operation #"+str(i)+": Start")
         print("Operation name: "+ops.name)
         #a convolutional layer
         if ops.name =='conv':
            #the convolution filter/kernel
            mat=ops.inputs['filter']
            for t in self.graph.operations:
               #find out which file is its data
               if t.outputs['output']==mat:
                  self.variablen[t.outputs['output']]=[] #t.outputs['output'] has form variable_xx
                  print("Reading weight data from "+t.attribs['label']+".dat ...")
                  ma=open(t.attribs['label']+'.dat','rb')
                  os.chdir("../3pxnet-compiler/autogen")
                  # Create C header file
                  # Replace dots with slashes
                  head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                  npi=self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),False,2,
                                           len(self.matrixmul)==0 and len(self.conv)==0)
                  # For dense, npi=0
                  if npi!=0:
                     print("Packs per kernel #: "+str(int(npi/32)))
                  # Store weight information
                  self.variablen[t.outputs['output']].append(npi)
                  self.variablen[t.outputs['output']].append(t)
                  self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                  self.variables[t.outputs['output']]=t.attribs['label']
                  ma.close()
                  os.chdir("..")
                  os.chdir(self.input_dir)
                  assert len(rank)==4
                  print("Padding: "+str(ops.attribs['padding'][0][0]))
                  #update current data shape
                  rank[3]=rank[3]+2*ops.attribs['padding'][0][0]-t.attribs['shape'][3]+1
                  rank[2] = rank[2] + 2 * ops.attribs['padding'][0][0] - t.attribs['shape'][2] + 1
                  rank[1] = t.attribs['shape'][0]
                  rank[0]=1
                  if ops.attribs['stride']!=[1,1]:
                     print("ERROR: current 3PXNet does not support stride")
                     exit()
                  break
            self.conv.append(i)
         #a fc layer
         elif ops.name=='matmul':
            #the kernel of fc layer
            mat=ops.inputs['B']
            for t in self.graph.operations:
               if t.outputs['output']==mat:
                  self.variablen[t.outputs['output']]=[]
                  ma=open(t.attribs['label']+'.dat','rb')
                  os.chdir("../3pxnet-compiler/autogen")
                  head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                  print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                  npi=self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),False,1,
                                           len(self.matrixmul)==0 and len(self.conv)==0)
                  if npi!=0:
                     print("Packs per kernel #: "+str(int(npi/32)))
                  self.variablen[t.outputs['output']].append(npi)
                  self.variablen[t.outputs['output']].append(t)
                  self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                  self.variables[t.outputs['output']]=t.attribs['label']
                  ma.close()
                  os.chdir("..")
                  os.chdir(self.input_dir)
                  assert len(rank)==2
                  rank[1]=t.attribs['shape'][1]
                  rank[0]=rank[0]
                  break
            self.matrixmul.append(i)
         #externally imported data, currently treated as input
         elif ops.name=='external':
            print("externally imported data, currently treated as input")
            print("WARNING: if it is not used as input, there will be errors.")
            self.in_shape=ops.attribs['shape']
            rank=self.in_shape.copy()
         # batch norm
         elif ops.name=='batch_normalization':
            mat=ops.inputs['mean']
            last=False
            #determine the last batch layer
            for t in self.graph.operations:
               if 'output' in t.outputs.keys() and t.outputs['output']==mat and t.attribs['label'].find(self.batch_last,0,-1) !=-1:
                  last=True
                  break
            print("Is the last batch normalization layer: "+str(last))
            #if it is the last batch norm layer, write out everything into the header file
            if last:
               for b in range(4):
                  if b ==0:
                     mat=ops.inputs['mean']
                  elif b ==1:
                     mat=ops.inputs['variance']
                  elif b ==2:
                     mat=ops.inputs['offset']
                  else :
                     mat=ops.inputs['scale']
                  for t in self.graph.operations:
                     if t.outputs['output']==mat:
                        assert t.attribs['shape'][0]==rank[0]
                        assert t.attribs['shape'][1] == rank[1]
                        self.variablen[t.outputs['output']]=[]
                        ma=open(t.attribs['label']+'.dat','rb')
                        os.chdir("../3pxnet-compiler/autogen")
                        print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                        head=open(self.replace_non_ascii(t.attribs['label'])+'.h','w+')
                        head.write("#define _"+self.replace_non_ascii(t.attribs['label'])+" {\\\n")
                        self.decoding_data(ma,head,self.replace_non_ascii(t.attribs['label']),True,0,False)
                        head.close()
                        self.source.write("#include \""+self.replace_non_ascii(t.attribs['label'])+'.h'+"\" \n")
                        self.variables[t.outputs['output']]=t.attribs['label']
                        ma.close()
                        os.chdir("..")
                        os.chdir(self.input_dir)
                        break
            #else, collect all four data needed to computer threshold and sign
            else:
               for b in range(4):
                  if b ==0:
                     mat=ops.inputs['mean']
                  elif b ==1:
                     mat=ops.inputs['variance']
                  elif b ==2:
                     mat=ops.inputs['offset']
                  else :
                     mat=ops.inputs['scale']
                  for t in self.graph.operations:
                     if t.outputs['output']==mat:
                        assert t.attribs['shape'][0] == rank[0]
                        assert t.attribs['shape'][1] == rank[1]
                        self.variablen[t.outputs['output']]=[]
                        ma=open(t.attribs['label']+'.dat','rb')
                        os.chdir("../3pxnet-compiler/autogen")
                        if b==0:
                           print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                           self.mean[t.outputs['output']]=self.decoding_data(
                              ma,None,self.replace_non_ascii(t.attribs['label']),False,0,False)
                           self.tempmean = t.outputs['output']
                        elif b==1:
                           print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                           self.var[t.outputs['output']]=self.decoding_data(
                              ma,None,self.replace_non_ascii(t.attribs['label']),False,0,False)
                           self.tempvar = t.outputs['output']
                        elif b==2:
                           print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                           self.beta[t.outputs['output']]=self.decoding_data(
                              ma,None,self.replace_non_ascii(t.attribs['label']),False,0,False)
                           self.tempbeta = t.outputs['output']
                        else :
                           print("Reading weight data from " + t.attribs['label'] + ".dat ...")
                           self.gamma[t.outputs['output']]=self.decoding_data(
                              ma,None,self.replace_non_ascii(t.attribs['label']),False,0,False)
                           self.tempgamma = t.outputs['output']
                        self.variables[t.outputs['output']]=t.attribs['label']
                        ma.close()
                        os.chdir("..")
                        os.chdir(self.input_dir)
                        break
            self.batchn[i]=[]
            self.batchn[i].append(ops.inputs['mean'])
            self.batchn[i].append(ops.inputs['variance'])
            self.batchn[i].append(ops.inputs['offset'])#bias,beta
            self.batchn[i].append(ops.inputs['scale'])#weight,gamma
         #if pooling
         elif ops.name=='max_pool':
            rank[3]=int(rank[3]/ops.attribs['size'][3])
            rank[2]=int(rank[2]/ops.attribs['size'][2])
            rank[1] = int(rank[1] / ops.attribs['size'][1])
            rank[0] = int(rank[0] / ops.attribs['size'][0])
         #clamp is considered as binarize output. If it is not used in this way, error would be given
         elif ops.name=='clamp':
            print("WARNING: clamp is considered as binarize output. If it is not used in this way, error would be given")
            if ops.inputs['a']!=-1 or ops.inputs['b']!=1:
               print("ERROR: 3PXNet inference library only holds 1 or -1.")
               exit()
         #current library does not have reshape function. so if the reshaped dimension is not found in the
         #current shape, then error would be given
         elif ops.name=='reshape':
            rank_flag=False
            temp=1
            for r in range(len(rank)):
               temp*=rank[r]
            for r in range(len(ops.attribs['shape'])):
               if temp==ops.attribs['shape'][r]:
                  rank_flag=True
            if not rank_flag:
               print("ERROR: current 3PXNet library does not support reshaping")
               exit()
            rank=ops.attribs['shape']
            for r in range(len(rank)):
               if rank[r]==-1:
                  rank[r]=1
         #softmax is always added at the end. If not, then the compiler would only give an error
         elif ops.name=='softmax':
            print("WARNING: current 3PXNet library does not support softmax function")
         #these three operations don't have impact on performance
         elif ops.name=='squeeze':
            for r in ops.attribs['axes']:
               del rank[r]
            print("Squeeze has no effect on inference, therefore it is skipped")
         elif ops.name=='unsqueeze' :
            for r in ops.attribs['axes']:
               rank.insert(r,1)
            print("Unsqueeze has no effect on inference, therefore it is skipped")
         elif ops.name=='variable':
            print('Define '+ops.attribs['label']+ ' as '+ops.outputs['output'])
            i+=1
            continue
         #same as softmax, if it's not used in the end, error would be given
         elif ops.name=='log' :
            if ops!=self.graph.operations[-1]:
               print("ERROR: current 3PXNet does not support log function")
               exit()
            else:
               print("WARNING: current 3PXNet does not support log function, but it doesn't affect the result")
         elif ops.name=='slice':
            rank[ops.attribs['axes'][0]]=ops.attribs['end'][0]
            print("WARNING: slice operation is skipped. If this is not operating on the input, the result will be wrong")
         else:
            print("ERROR: current 3PXNet does not support "+ops.name+" function")
            exit()
         print("Operation output shape:",end=' ')
         print(rank)
         i+=1

   def write_last_layer(self):
      if len(self.tempweight) != 0:
         tt = np.copy(self.tempweight)
         if len(tt.shape) == 4:
            self.writecn(True, [tt.shape[3], tt.shape[2], tt.shape[1], tt.shape[0]],
                              self.tempweight, self.tempsparse, self.tempoutput, self.name)
         else:
            self.writefc(True, tt.shape, self.tempweight, self.tempsparse, self.tempoutput,
                              self.name)

   def calculate_batch(self):
      '''
         Calculate batch normalization threshold and signs

      :return: NA
      '''
      os.chdir("../3pxnet-compiler/autogen")
      if len(self.var.keys()) != len(self.mean.keys()) or len(self.var.keys()) != len(self.gamma.keys()) or \
              len(self.var.keys()) != len(self.beta.keys()):
         print("error with batch normalization number")
         exit()
      thresh={}
      sign={}
      k=0
      #calculate threshold and sign
      #variables: map a variable_# to its name stored in nnef directory
      for i in self.batchn.keys():
         if self.variables[self.batchn[i][0]][0:self.search_non_ascii(self.variables[self.batchn[i][0]])] == self.batch_last:
            continue
         temp=bitarray()
         epsilon=[self.graph.operations[i].attribs['epsilon']]*len(self.var[self.batchn[i][1]])
         thresh[k]=[]
         sign[k]=[]
         for j in range(len(self.var[self.batchn[i][1]])):
            thresh[k].append(self.mean[self.batchn[i][0]][j]-np.sqrt(self.var[self.batchn[i][1]][j]+epsilon[j])/
                             self.gamma[self.batchn[i][3]][j]*self.beta[self.batchn[i][2]][j])
         for j in range(len(self.var[self.batchn[i][1]])):
            temp.append(int(self.gamma[self.batchn[i][3]][j]>0))
            if j%32 == 31 :
               sign[k].append(str("0x%x" % int(temp.to01(), 2)))
               temp=bitarray()
         head=open("bn"+str(k+1)+".h",'w+')
         head.write("#define bn"+str(k+1)+"_thresh {\\\n")
         for j in range(len(self.var[self.batchn[i][1]])):
            head.write(str(thresh[k][j])+", \\\n")
         head.write("} \n")
         head.write("#define bn"+str(k+1)+"_sign {\\\n")
         for j in range(int(len(self.var[self.batchn[i][1]])/32)):
            head.write(str(sign[k][j])+", \\\n")
         head.write("} \n")
         self.source.write("#include \"bn"+str(k+1)+".h\" \n")
         k+=1

   def testsparse(self,i,total_ops):
      '''
         Test sparsity.
         The way to do this is to search for the word "indices" in a given header file

      :param i: index of current operation in "total_ops" list
      :param total_ops: a list of indices of all matrix multiplication/convolution in graph.operations
      :return: whether this layer is sparse or not
      '''
      if 'filter' in self.graph.operations[total_ops[i]].inputs:
         test_sparse = open(
            self.replace_non_ascii(self.variables[self.graph.operations[total_ops[i]].inputs['filter']]) + '.h', 'r')
         sparse = test_sparse.read().find("indices", 0, -1)
      else:
         test_sparse = open(self.replace_non_ascii(self.variables[self.graph.operations[total_ops[i]].inputs['B']]) + '.h',
                            'r')
         sparse = test_sparse.read().find("indices", 0, -1)
      if sparse == -1:
         sparse = False
      else:
         sparse = True
      test_sparse.close()
      return sparse

   def write_source_second(self):
      '''
      Write out all remaining source code
      It can be considered as two parts: the first one writes out all specifications of one layer, such as its input
      size, kernel size, and output size. For a convolutional layer, padding and pooling information are also defined
      in this part. Besides, batch normalization information is defined in this part as well.
      The second part writes out used functions defined in inference engine according to different layer settings.

      :return: NA
      '''
      #write source file: first part
      print("-----------------------------------------")
      self.source.write('#include \"image.h\"\n')
      self.source.write("static uint8_t l1_act[] = IMAGES ; \n")
      self.source.write("static uint8_t   labels[] = LABELS; \n")
      total_ops=self.matrixmul+self.conv
      total_ops=sorted(total_ops)
      for i in range(1, len(total_ops) + 1):
         sparse=self.testsparse(i-1,total_ops)
         #fc layer
         if total_ops[i-1] in self.matrixmul:
            # number of inputs
            self.source.write("#define F"+str(i)+"I  "+str(
               self.variablen[self.graph.operations[total_ops[i-1]].inputs['B']][1].attribs['shape'][0])+"\n")
            #unpruned inputs
            self.source.write("#define F"+str(i)+"NPI  "+str(
               self.variablen[self.graph.operations[total_ops[i-1]].inputs['B']][0])+"\n")
            #number of outputs
            self.source.write("#define F"+str(i)+"O  "+str(
               self.variablen[self.graph.operations[total_ops[i-1]].inputs['B']][1].attribs['shape'][1])+"\n")
            #weight data
            self.source.write("static pckDtype l"+str(i)+"wght[] = _"+self.replace_non_ascii(
               self.variablen[self.graph.operations[total_ops[i-1]].inputs['B']][1].attribs['label'])+" ;\n")
            if sparse:
               self.source.write("static uint8_t l"+str(i)+"ind[] = _"+self.replace_non_ascii(
                  self.variablen[self.graph.operations[total_ops[i-1]].inputs['B']][1].attribs['label'])+"_indices ;\n")
            #previous layer is fc
            if i!=1 and total_ops[i-2] in self.matrixmul:
               self.source.write("static pckDtype l"+str(i)+"act_bin[F"+str(i-1)+"O/pckWdt]; \n")
            #first layer
            else:
               self.source.write("static pckDtype l"+str(i)+"act_bin[F"+str(i)+"I/pckWdt]; \n")
         #conv layer
         else:
            self.source.write("#define C"+str(i)+"KXY "+str(
               self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][1].attribs['shape'][2])+"\n")
            #input of the first layer
            if i == 1:
               #input number will change according to input size
               self.source.write("#define C1XY   "+str(int(self.in_shape[2]))+"\n")
               self.source.write("#define C1Z   "+str(self.in_shape[1])+"\n")
               self.source.write("#define C"+str(i)+"KZ "+str(
                  self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][1].attribs['shape'][0])+"\n")
            else:
               #determine the XY dimension from pervious layer's padding and pooling number
               self.source.write('#define C'+str(i)+"XY ((2*C"+str(i-1)+"PD+C"+str(i-1)+"XY-C"+str(i-1)+"KXY+1)/C"+
                                 str(i-1)+"PL) \n" )
               #determine the Z dimension for activation
               self.source.write("#define C"+str(i)+"Z "+str(
                  self.variablen[self.graph.operations[total_ops[i-2]].inputs['filter']][1].attribs['shape'][0])+'\n')
               #determine kernel's Z dimension
               self.source.write("#define C"+str(i)+"KZ "+str(
                  self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][1].attribs['shape'][0])+"\n")
               #size of activation
               self.source.write("static pckDtype l"+str(i)+"act_bin[C"+str(i)+"XY*C"+str(i)+"XY*C"+str(i)+"Z/pckWdt]; \n")
            #size of padding
            self.source.write("#define C"+str(i)+"PD "+str(self.graph.operations[total_ops[i-1]].attribs['padding'][0][0])+'\n')
            #determine pooling
            #use "look ahead" method
            if i != len(total_ops):
               #search between two matrix operations for max pooling function
               for p in range(total_ops[i-1],total_ops[i]):
                  if self.graph.operations[p].name=='max_pool':
                     self.source.write("#define C"+str(i)+"PL "+str(self.graph.operations[p].attribs['size'][3])+'\n')
                     break
                  if p == total_ops[i]-1:
                     self.source.write("#define C"+str(i)+"PL 1 \n")
            #the last layer
            else:
               for p in range(total_ops[i-1],len(self.graph.operations)):
                  if self.graph.operations[p].name=='max_pool':
                     self.source.write("#define C"+str(i)+"PL "+str(self.graph.operations[p].attribs['size'][3])+'\n')
                     break
                  if p == len(self.graph.operations)-1:
                     self.source.write("#define C"+str(i)+"PL 1 \n")
            #unpruned inputs
            if self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][0] != 0:
               self.source.write("#define C"+str(i)+"NPI "+str(self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][0])+'\n')
               if sparse:
                  self.source.write("static uint8_t l"+str(i)+"ind[] = _"+self.replace_non_ascii(
                     self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][1].attribs['label'])+"_indices ;\n")
            if i == 1:
               self.source.write("static int8_t l"+str(i)+"wght[] = _"+self.replace_non_ascii(
                  self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][1].attribs['label'])+" ;\n")
            else:
               self.source.write("static pckDtype l"+str(i)+"wght[] = _"+self.replace_non_ascii(
                  self.variablen[self.graph.operations[total_ops[i-1]].inputs['filter']][1].attribs['label'])+" ;\n")

      #theoretically the output can be other number, we pick 10 right now
      self.source.write("static float output[10]; \n")

      number=0
      #write batch norm
      for i in self.batchn.keys():
         number=len([k for k in self.batchn.keys() if k < i])+1
         #the last batch norm layer
         if number == len(self.batchn.keys()):
            self.source.write("static bnDtype bn"+str(number)+"mean[] = _"+self.replace_non_ascii(self.variables[self.batchn[i][0]])+" ; \n")
            self.source.write("static bnDtype bn"+str(number)+"var[] = _"+self.replace_non_ascii(self.variables[self.batchn[i][1]])+" ; \n")
            self.source.write("static bnDtype bn"+str(number)+"gamma[] = _"+self.replace_non_ascii(self.variables[self.batchn[i][3]])+" ; \n")
            self.source.write("static bnDtype bn"+str(number)+"beta[] = _"+self.replace_non_ascii(self.variables[self.batchn[i][2]])+" ; \n")
         #otherwise use only threshold and sign
         else:
            self.source.write("static bnDtype bn"+str(number)+"thr[] = bn"+str(number)+"_thresh ; \n")
            self.source.write("static pckDtype bn"+str(number)+"sign[] = bn"+str(number)+"_sign ; \n")


      self.source.write("int main(){ \n\tint correct = 0; \n\tfor(int img = 0; img < "+str(int(self.test_end_id-self.test_start_id))+
                        "; img++) {\n\t\tuint8_t *curr_im = l1_act + img*")

      #fc network
      if len(self.conv)==0:
         self.source.write("784*sizeof(uint8_t);\n\t\t")
         self.source.write("packBinThrsArr(curr_im, l1act_bin, F1I, 1);\n\t\t")
      #mixed
      else:
         self.source.write(str(self.in_shape[2])+"*"+str(self.in_shape[3])+"*"+str(self.in_shape[1])+"*sizeof(uint8_t);\n\t\t")

      # write out source code: second part
      for i in range(len(total_ops)):
         sparse=self.testsparse(i,total_ops)
         print("Operation ", end='')
         if i < len(total_ops) - 1:
            for r in range(total_ops[i], total_ops[i + 1]):
               if self.graph.operations[r].name == 'unsqueeze' or self.graph.operations[r].name == 'squeeze' or \
                       self.graph.operations[r].name == 'clamp' or self.graph.operations[
                  r].name == 'batch_normalization' or self.graph.operations[r].name=='max_pool':
                  print('# ' + str(r), end=' ')
         else:
            for r in range(total_ops[i], len(self.graph.operations)):
               if self.graph.operations[r].name == 'unsqueeze' or self.graph.operations[r].name == 'squeeze' or \
                       self.graph.operations[r].name == 'clamp' or self.graph.operations[
                  r].name == 'batch_normalization' or self.graph.operations[r].name=='max_pool':
                  print('# ' + str(r), end=' ')
         print("uses C library function ", end='')
         if total_ops[i] in self.matrixmul:
            #normal layer with batchnorm
            if i<len(total_ops)-1 and (total_ops[j] in self.batchn.keys() for j in range(total_ops[i],total_ops[i+1])):
               if sparse:
                  print("Fc3pxnWrap")
                  self.source.write("Fc3pxnWrap(l"+str(i+1)+"act_bin, l"+str(i+1)+"wght, l"+str(i+1)+"ind, F"+str(i+1)+"NPI, F"+str(i+1)+"O, l"+str(i+2)+"act_bin, bn"+str(i+1)+"thr, bn"+str(i+1)+"sign);\n\t\t")
               else:
                  print("FcXnorWrap")
                  self.source.write("FcXnorWrap(l" + str(i + 1) + "act_bin, l" + str(i + 1) + "wght, F" + str(i + 1) + "I, F" + str(i + 1) + "O, l" + str(
                     i + 2) + "act_bin, bn" + str(i + 1) + "thr, bn" + str(i + 1) + "sign);\n\t\t")
            #normal layer without batchnorm
            elif i<len(total_ops)-1:
               if sparse:
                  print("Fc3pxnWrap")
                  self.source.write("Fc3pxnWrap(l"+str(i+1)+"act_bin, l"+str(i+1)+"wght, l"+str(i+1)+"ind, F"+str(i+1)+"NPI, F"+str(i+1)+"O, l"+str(i+2)+"act_bin, NULL, NULL);\n\t\t")
               else:
                  print("FcXnorWrap")
                  self.source.write("FcXnorWrap(l" + str(i + 1) + "act_bin, l" + str(i + 1) + "wght, F" + str(i + 1) + "I, F" + str(i + 1) + "O, l" + str(
                     i + 2) + "act_bin, NULL, NULL);\n\t\t")
            #last layer
            elif i==len(total_ops)-1 and (total_ops[j] in self.batchn.keys() for j in range(total_ops[i],len(self.graph.operations))):
               if sparse:
                  print("Fc3pxnNoBinWrap")
                  self.source.write("int res = Fc3pxnNoBinWrap(l"+str(i+1)+"act_bin, l"+str(i+1)+"wght, l"+str(i+1)+"ind, F"+str(i+1)+"NPI, F"+str(i+1)+"O, output, bn")
                  self.source.write(str(i+1)+"mean, bn"+str(i+1)+"var, bn"+str(i+1)+"gamma, bn"+str(i+1)+"beta);\n\t\t")
               else:
                  print("FcXnorNoBinWrap")
                  self.source.write(
                     "int res = FcXnorNoBinWrap(l" + str(i + 1) + "act_bin, l" + str(i + 1) + "wght, F" + str(i + 1) + "I, F" + str(i + 1) + "O, output, bn")
                  self.source.write(str(i + 1) + "mean, bn" + str(i + 1) + "var, bn" + str(i + 1) + "gamma, bn" + str(
                     i + 1) + "beta);\n\t\t")
         else:
            #first layer. should be BWN
            #with batchnorm
            if i==0 and (total_ops[j] in self.batchn.keys() for j in range(total_ops[i],total_ops[i+1])):
               print("CnBnBwn")
               self.source.write("CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);\n\t\t")
            #without batch norm
            elif i==0:
               print("CnBnBwn")
               self.source.write("CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, NULL,NULL)\n\t\t")
            #normal layer
            elif i<len(total_ops)-1 and (total_ops[j] in self.batchn.keys() for j in range(total_ops[i],total_ops[i+1])):
               if sparse:
                  print("Cn3pxnWrap")
                  self.source.write("Cn3pxnWrap(l"+str(i+1)+"act_bin, l"+str(i+1)+"wght, l"+str(i+1)+"ind, C"+str(i+1)+"NPI, C"+str(i+1))
                  self.source.write("Z, C"+str(i+1)+"XY, C"+str(i+1)+"XY, C"+str(i+1)+"Z, C"+str(i+1)+"KXY, C"+str(i+1)+"KXY, C"+str(i+1)+"KZ, l"+str(i+2)+"act_bin, C"+str(i+1))
                  self.source.write("PD, C"+str(i+1)+"PL, bn"+str(i+1)+"thr, bn"+str(i+1)+"sign);\n\t\t")
               else:
                  print("CnXnorWrap")
                  self.source.write(
                     "CnXnorWrap(l" + str(i + 1) + "act_bin, l" + str(i + 1) + "wght, C" + str(i + 1))
                  self.source.write("Z, C" + str(i + 1) + "XY, C" + str(i + 1) + "XY, C" + str(i + 1) + "Z, C" + str(
                     i + 1) + "KXY, C" + str(i + 1) + "KXY, C" + str(i + 1) + "KZ, l" + str(i + 2) + "act_bin, C" + str(
                     i + 1))
                  self.source.write("PD, C" + str(i + 1) + "PL, bn" + str(i + 1) + "thr, bn" + str(i + 1) + "sign);\n\t\t")
            #last layer
            elif i == len(total_ops)-1:
               if sparse:
                  print("Cn3pxnNoBinWrap")
                  self.source.write("int res = Cn3pxnNoBinWrap(l"+str(i+1)+"act_bin, l"+str(i+1)+"wght, l"+str(i+1)+"ind, C"+str(i+1)+"NPI, C"+str(i+1))
                  self.source.write("Z, C"+str(i+1)+"XY, C"+str(i+1)+"XY, C"+str(i+1)+"Z, C"+str(i+1)+"KXY, C"+str(i+1)+"KXY, C"+str(i+1)+"KZ, output, C"+str(i+1))
                  self.source.write("PD, C"+str(i+1)+"PL, bn"+str(i+1)+"mean, bn"+str(i+1)+"var, bn"+str(i+1)+"gamma, bn"+str(i+1)+"beta);\n\t\t")
               else:
                  print("CnXnorNoBinWrap")
                  self.source.write("int res = CnXnorNoBinWrap(l" + str(i + 1) + "act_bin, l" + str(i + 1) + "wght, C" + str(i + 1))
                  self.source.write("Z, C" + str(i + 1) + "XY, C" + str(i + 1) + "XY, C" + str(i + 1) + "Z, C" + str(
                     i + 1) + "KXY, C" + str(i + 1) + "KXY, C" + str(i + 1) + "KZ, output, C" + str(i + 1))
                  self.source.write("PD, C" + str(i + 1) + "PL, bn" + str(i + 1) + "mean, bn" + str(i + 1) + "var, bn" + str(
                     i + 1) + "gamma, bn" + str(i + 1) + "beta);\n\t\t")
            #without batch norm
            else:
               if sparse:
                  print("Cn3pxnWrap")
                  self.source.write("Cn3pxnWrap(l"+str(i+1)+"act_bin, l"+str(i+1)+"wght, l"+str(i+1)+"ind, C"+str(i+1)+"NPI, C"+str(i+1))
                  self.source.write("Z, C"+str(i+1)+"XY, C"+str(i+1)+"XY, C"+str(i+1)+"Z, C"+str(i+1)+"KXY, C"+str(i+1)+"KXY, C"+str(i+1)+"KZ, l"+str(i+2)+"act_bin, C"+str(i+1))
                  self.source.write("PD, C"+str(i+1)+"PL, NULL, NULL);\n\t\t")
               else:
                  print("CnXnorWrap")
                  self.source.write(
                     "CnXnorWrap(l" + str(i + 1) + "act_bin, l" + str(i + 1) + "wght, C" + str(i + 1))
                  self.source.write("Z, C" + str(i + 1) + "XY, C" + str(i + 1) + "XY, C" + str(i + 1) + "Z, C" + str(
                     i + 1) + "KXY, C" + str(i + 1) + "KXY, C" + str(i + 1) + "KZ, l" + str(i + 2) + "act_bin, C" + str(
                     i + 1))
                  self.source.write("PD, C" + str(i + 1) + "PL, NULL, NULL);\n\t\t")
         print("-----------------------------------------")
      #testing and inference
      self.source.write("float max = -INFINITY; \n\t\tint maxIdx = 0; \n\t\tfor (int i = 0; i <10; i++) { \n\t\t\t printf(\"%f, \", output[i]);\n\t\t\t if (output[i] > max) { \n\t\t\t\t max = output[i]; \n\t\t\t\t")
      self.source.write("maxIdx = i;\n\t\t\t }\n\t\t}\n\t\t")
      self.source.write("printf(\"\\n\");")
      self.source.write("printf(\"Image %d: label: %d, actual: %d\\n\",img, maxIdx, labels[img]); \n\t\t")
      self.source.write("if (maxIdx == labels[img]) correct += 1; \n\t}\n\tprintf(\"Accuracy: %f%%\\n\", 100.0*(float)correct/"
                        +str(int(self.test_end_id-self.test_start_id))+"); \n\treturn (EXIT_SUCCESS); \n}")
      self.source.close()


   def write_images(self):
      '''
         write out images for both testing and inference

      :return: NA
      '''
      image=open('image.h','w+')
      os.chdir('..')
      from shutil import copyfile
      copyfile('../3pxnet-training/utils_own.py','utils_own.py')
      copyfile('../3pxnet-training/utils.py', 'utils.py')
      import torch
      from utils_own import load_dataset
      '''
      Imported from training engine
      Give training/testing data loader and class information for several image datasets
      '''
      trainset, testset, classes = load_dataset(self.dataset)
      testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=2)
      label=testloader.dataset.targets[self.test_start_id:self.test_end_id]
      if self.dataset=='MNIST':
         testdata=torch.tensor([2,2,2,2])
         #testdata.cuda()
         testdata=testloader.dataset.data[self.test_start_id:self.test_end_id,:,:]
         testdata = torch.reshape(testdata, (self.test_end_id-self.test_start_id, 784))
         rank=testdata.shape
         temp_array = testdata.clone()
         if len(self.list)!=0:
            for i in range(768):
               temp_array[0:, i] = testdata[:, self.list[0][i]]
         testdata = temp_array
         image.write('#define IMAGES {\\\n')
         for n in range(self.test_end_id-self.test_start_id):
            for y in range(28):
               for x in range(28):
                  image.write(str(int(testdata[n][y*28+x].item()>0))+', ')
               image.write('\\\n')
            image.write('\\\n')
         image.write('}\n')
         image.write('#define LABELS {\\\n')
         for i in range(self.test_end_id-self.test_start_id):
            image.write(str(label[i].item())+', ')
         image.write('}\n')
         image.close()
      else:
         testdata=torch.from_numpy(testloader.dataset.data[self.test_start_id:self.test_end_id,:,:,:]).permute(0,3,1,2)
         #testdata.cuda()
         rank = testdata.shape
         image.write('#define IMAGES {\\\n')
         for n in range(self.test_end_id-self.test_start_id):
            for y in range(rank[2]):
               for x in range(rank[3]):
                  for z in range(rank[1]):
                     image.write(str(int(testdata[n][z][y][x].item()))+', ')
            image.write('\\\n')
         image.write('}\n')
         image.write('#define LABELS {\\\n')
         for i in range(self.test_end_id-self.test_start_id):
            image.write(str(label[i])+', ')
         image.write('}\n')
         image.close()

def main():
   print("WARNING: Current 3PXNet inference library does not support operations "
         "other than convolution or matrix multiplication")
   print("All other operations will be skipped.")
   # Argument parsing
   parser = argparse.ArgumentParser(description='Automatically generate inference code')
   parser.add_argument('--input', help="""Name of input directory. This should be the converted NNEF "
      "formatted neural network which ends with .nnef with no other modifications. Example: --input=FC_Small.nnef""")
   parser.add_argument('--dataset', metavar='DATASET', default='MNIST',
                       help='Dataset to test on. Currently choose from MNIST and CIFAR10')
   parser.add_argument('--test_start_id',default=0,help='The starting index of dataset for testing')
   parser.add_argument('--test_end_id', default=100, help='The ending index of dataset for testing')
   args = parser.parse_args()
   dataset = args.dataset
   input_dir = args.input
   test_start_id=args.test_start_id
   test_end_id=args.test_end_id

   converter=convert(input_dir,dataset,test_start_id,test_end_id)
   #load the nnef graph into compiler
   converter.loadgraph()
   # the last layer will not be binarized, so its batch norm has to be dealt differently
   # this function finds such batch norm operation
   converter.find_batch_last()
   # write included headers into source code
   converter.write_source_first()
   # for each operation shown in the graph, compile it
   converter.processing_graph()
   # the last layer should be written out as well
   converter.write_last_layer()
   # calculate batch normalization threshold and sign
   converter.calculate_batch()
   # write out all remaining source code
   converter.write_source_second()
   # write out images for both inference and testing
   converter.write_images()

if __name__ == '__main__':
   main()
