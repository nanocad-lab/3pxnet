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

"""Tests XNOR/3PXnet implementation against reference
Author:        Wojciech Romaszkan
Organization:  NanoCAD Laboratory, University of California, Los Angeles
License:       MIT
"""

import subprocess

__author__ = "Wojciech Romaszkan, NanoCAD Laboratory, UCLA"
__license__ = "MIT"


class runValidation(object):

   def __init__(self):
      self.f = open("logfile", "w")
      # Iterations
      self.iters = 100
      # Layers to run
      self.layers = [" -f ", " -c ", " -c -d ", " -c -l 2 ", " -c -l 2 -d "]
      # batch norm
      self.bnorm = [" ", " -b "]
      # output binarization
      self.outbin = [" ", " -n "]
      # sparsity
      self.sparse = [" ", " -s 90 -p "]

   def run(self):

      # Failure counter
      fail = 0

      for layer in self.layers:
         for bn in self.bnorm:
            for ob in self.outbin:
               for sp in self.sparse:
                  cmdString = "./validation" + layer + ob + bn + sp + " -i " + str(self.iters)
                  print("Running: " + cmdString)
                  result = subprocess.call(cmdString, shell=True, stdout=self.f)
                  if result:
                     print("FAILED")
                     fail = 1
                  else:
                     print("PASSED")
      # Check if any of the tests failed   
      if fail:
         print("Some tests failed")
      return True


def main():
   validator = runValidation()
   validator.run()

if __name__ == '__main__':
   main()
