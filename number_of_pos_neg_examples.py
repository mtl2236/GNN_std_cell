import sys
import os
import string

pos_num=0
neg_num=0
f=open('all.txt','r')
all_file=f.readlines()
for i in range(len(all_file)):
    if('y=torch.tensor([1])' in all_file[i]):
        pos_num=pos_num+1
    if('y=torch.tensor([0])' in all_file[i]):
        neg_num=neg_num+1
print(pos_num)   
print(neg_num)     