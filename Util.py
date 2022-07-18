
import os
import gc
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split as TTS
import sys
import timm
import pickle
from torch.utils.data import random_split,DataLoader

def img_paths_list(root_dir):
    root_list = glob.glob(root_dir)
    class_map = {}
    class_distribution = {}
    
    for img_path in root_list:
        class_name = img_path.split(os.sep)[-2]
        if class_name not in class_distribution:
            class_distribution[class_name] = 1
        else:
            class_distribution[class_name] +=1
                
    for index, entity in enumerate(class_distribution):
        class_map[entity] = index
    print("Dataset Distribution:\n")
    
    print(class_distribution)
    print("\n\nClass indices:\n")
    print(class_map)

    data = []
    for img_pth in tqdm(root_list):
        class_name = img_pth.split(os.sep)[-2]
        data.append([img_pth, class_name])
        
    return data, class_map


def data_split(img_paths):
    X = []
    y = []
    for img_path in img_paths:
        X.append(img_path[0])
        y.append(img_path[1])
    
    X_train_list, X_test_list, y_train_list, y_test_list = TTS(X, y, stratify = y, test_size=0.1)
    
    train_img_paths = []
    test_img_paths = []
    for i in range(0, len(X_train_list)):
        train_img_paths.append([X_train_list[i], y_train_list[i]])
    for i in range(0, len(X_test_list)):
        test_img_paths.append([X_test_list[i], y_test_list[i]])
        
    return train_img_paths, test_img_paths



def create_transforms():
    return A.Compose(
            [
                A.CenterCrop(height = 100, width = 100, p=1.0),
                ToTensorV2()
            ]
    )


def split_train_valid_test(train_dataset,test_set):
    m = len(train_dataset)
    test_split_size = 0.1

    print("Total training data: " + str(m))

    try:
        train_set,val_set=random_split(train_dataset,[int(m-m*test_split_size),int(m*test_split_size)])
    except:
        train_set,val_set=random_split(train_dataset,[int(m-m*test_split_size),int(m*test_split_size+1)])
    
    print("len of train,valid,test: ",len(train_set), len(val_set), len(test_set))
    return train_set,val_set



def load_data(train_set,val_set,test_set,batch_size):
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
    print("len of train,val,test loaders:",len(train_loader),len(val_loader),len(test_loader))
    return train_loader,val_loader,test_loader