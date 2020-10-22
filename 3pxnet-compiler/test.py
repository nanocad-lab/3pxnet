import numpy as np
import os
Golden=open('../3pxnet-compiler/__Golden.txt')
Golden=Golden.readlines()
if len(Golden)==4:
   print(Golden[0],end='')
   print(Golden[1],end='')
   print(Golden[2],end='')
   print("Output labels are being compared.")
   Golden=np.fromstring(Golden[3],dtype=int,sep=' ')
else:
   print(Golden[0], end='')
   print("Output labels are being compared.")
   Golden = np.fromstring(Golden[1], dtype=int, sep=' ')
f=open('../3pxnet-compiler/__Compiled.txt')
contents=f.readlines()
Tested=[]
for f1 in contents:
   index=f1.find('label')
   Tested.append(int(f1[index+7]))
Tested=np.array(Tested)
failed=0
for i in range(len(Tested)):
   if Golden[i]!=Tested[i]:
      failed+=1
   print("Test id: "+str(i)+"    Golden: "+str(Golden[i])+"   Tested: "+str(Tested[i]))
if failed>=5:
   print("Test failed")
else:
   print("Test passed")