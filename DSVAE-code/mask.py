import os
import shutil
import sys
import random
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf)


def mask_percent(train_arr, train_mask1,train_mask2, ratio):
    j = 0
    for intr in train_arr: 
       # print(intr,type(intr))           
        ind = np.transpose(np.nonzero(intr))
       # print(ind,type(ind))
        ind  = np.reshape(ind,(-1))
        sam_ind1 = np.random.choice(ind, int(ratio*ind.shape[0]))#,replace=False bu fang hui£¬defult=true
        sam_ind2 = np.random.choice(ind, int(ratio*ind.shape[0]))
       # print(sam_ind1,sam_ind2)
       # print(train_mask1[j])
        train_mask1[j][sam_ind1] = 0 
        train_mask2[j][sam_ind2] = 0                          
        j+=1      
    return train_mask1, train_mask2


def mask_for_predict(train_arr, train_mask,n_items):
    k = 1
    gt_list = []
    for i in range(k):
        j = 0
        gt_arr = np.zeros((train_arr.shape[0],n_items),dtype=np.float32)
        for intr in train_arr:
            
            ind = np.transpose(np.nonzero(intr))
            ind  = np.reshape(ind,(-1))
            if ind.shape[0] > 34:
                sam_ind = np.random.choice(ind, int(0.15*ind.shape[0]))#0.15 34
                sub_sam_ind = np.random.choice(ind, int(0.8*sam_ind.shape[0]))
                train_mask[j][sub_sam_ind] = 0 
                gt_arr[j][sam_ind] = 1                           
            j+=1      
        gt_list.append(gt_arr)
    gt_list = np.concatenate(gt_list,0)

    return train_mask, gt_list