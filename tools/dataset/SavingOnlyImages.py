import numpy as np
import math
import h5py
from PIL import Image, ImageOps
import re
import json
import glob, os, os.path, shutil
from os import listdir
from os.path import isfile, join


def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path
        
again = 1



while again == 1:
    datasetarray = []
    folderorfile = input("Do you want to run a folder or just one file (folder/file/none)\n")
    if folderorfile == "folder":
        path = input("Folder name?\n")
        for file in os.listdir(path):
            if file.endswith(".hdf5"):
                print(os.path.join(path,file))
                datasetarray.append(os.path.join(path,file))
    elif folderorfile == "file":
        filename = input("What is the name of the file? (include .hdf5)\n")
        datasetarray.append(filename)
    elif folderorfile == "none":
        again = 0
        quit
    else:
        print("Invalid input")
        exit
    

    ## -------------------------- Extract Images and Steering Data -----------------------
    for file in datasetarray:
        h = []
        i = 0
        foundImg = 0
        foundSteering = 0
        dict = {}
        if (os.path.isdir(file[:-5]) == False):
            os.mkdir(file[:-5])
        else:
            shutil.rmtree(file[:-5])
            os.mkdir(file[:-5])
        with h5py.File(file, 'r') as f:
            for dset in traverse_datasets(f):
                if f[dset].shape==(240, 320, 3):
                    j = np.array(f[dset][:])
                    array = np.reshape(j, (240, 320,3))
                    im = Image.fromarray(array)
                    im = ImageOps.flip(im)
                    file_name = str(i)
                    file_name = file_name.zfill(5)
                    file_name = file[:-5] + "/" + file_name + ".jpg"
                    im.save(file_name)
                    i = i+1
                    foundImg = 1
                if 'steering' in dset:
                    h.append(np.array(f[dset]))
                    dict[i-1] = np.array(f[dset])
                    foundSteering = 1
                if (foundImg == 1 and foundSteering == 1):
                    foundImg = 0
                    foundSteering = 0
