
import os
import numpy as np
import random
import shutil

root = '.\\data\\coco\\images'
li = [3, 4, 5, 6, 7, 11, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 30, 34, 36, 37, 39, 42, 43, 66, 73, 74, 76]
perc = [0.8, 0.1, 0.1] # train, val, test


def split_by_pz(pz,mode):
    dest_dir = root + '\\' + mode
    
    rmake_dir(dest_dir)
    
    for index in pz:
        dir = root + '\\pz' + str(index)
        
        for filename in os.listdir(dir):
            f = os.path.join(dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                shutil.copy(f, dest_dir + '\\' + filename)

#Splitta direttamente i pazienti in train, val e test
def split_pz():
    random.seed(1)
    random.shuffle(li)
    pz_train, pz_val, pz_test = np.split(li, [int(len(li)*perc[0]), int(len(li)*(perc[0] + perc[1]))])
    #split_by_pz(pz_train,'train')
    split_by_pz(pz_val,'validation')
    split_by_pz(pz_test,'test')

#splitta le foto di ciascun paziente 80 in train, 10 val e 10 test
def split_by_perc():
    train_dir = root + '\\' + 'train'
    val_dir = root + '\\' + 'validation'
    test_dir = root + '\\' + 'test'
    
    rmake_dir(train_dir)
    rmake_dir(val_dir)
    rmake_dir(test_dir)
    
    random.seed(1)
    
    for index in li:
        src_dir = root + '\\pz' + str(index)
        pz_files = os.listdir(src_dir)
        random.shuffle(pz_files)
        pz_train, pz_val, pz_test = np.split(pz_files, [int(len(pz_files)*perc[0]), int(len(pz_files)*(perc[0] + perc[1]))])
       
        #copy_file(pz_train,src_dir,'train')
        copy_file(pz_val,src_dir,'validation')
        copy_file(pz_test,src_dir,'test')


def copy_file(pz_files,dir, mode):
    dest_dir = root + '\\' + mode
    
    for filename in pz_files:
        f = os.path.join(dir, filename)
    # checking if it is a file
        if os.path.isfile(f):
            shutil.copy(f, dest_dir + '\\' + filename)

def rmake_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    
if __name__ == "__main__":
    split_by_perc()