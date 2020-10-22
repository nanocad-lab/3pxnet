#!/usr/bin/env python

################################################################################ 
# MIT License
# 
# Copyright (c) 2019 UCLA NanoCAD Laboratory 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################ 

"""Generates MNIST header files for provided examples
Author:        Wojciech Romaszkan
Organization:  NanoCAD Laboratory, University of California, Los Angeles
License:       MIT
"""


import numpy as np
import idx2numpy as i2n

__author__ = "Wojciech Romaszkan, NanoCAD Laboratory, UCLA"
__license__ = "MIT"

class genMnistHeader(object):

   def __init__(self):
      # Images are concatenated into a single variable, so
      # the pointer needs to move by 784 from image to image

      # header file
      self.hdrFile = "mnist_img_lbl_norm.h"
      self.hdr = open(self.hdrFile, "w+") 

      # Number of images to copy
      self.numIm = 100

      # Starting index
      self.idxSt = 0

      # Normalization
      self.norm = False

   def run(self):
      # Get images and labels
      img = i2n.convert_from_file('t10k-images-idx3-ubyte')
      lbl = i2n.convert_from_file('t10k-labels-idx1-ubyte')

      self.hdr.write("#define MNIST_IMAGES {\\\n")
      for image in range(self.numIm):
         for row in range(28):
            for col in range(28):
               if self.norm:
                  temp = float(img[self.idxSt+image][row][col])/255.0
                  temp -= 0.1307
                  temp = temp/0.3015
                  if temp >= 0:
                     temp = 1
                  else:
                     temp = 0
                  self.hdr.write(str(temp) + ", ")
               else:
                  self.hdr.write(str(img[self.idxSt+image][row][col]) + ", ")
         self.hdr.write("\\\n")
      self.hdr.write("}\n\n")


      self.hdr.write("#define MNIST_LABELS {\\\n")
      for image in range(self.numIm):
         self.hdr.write(str(lbl[self.idxSt+image]) + ", ")
         self.hdr.write("\\\n")
      self.hdr.write("}\n\n")

def main():
   headGen = genMnistHeader()
   headGen.run()

if __name__ == '__main__':
   main()
