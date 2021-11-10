import numpy as np
import h5py
from PIL import Image, ImageOps
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re
import json
import os, os.path, shutil
from os import listdir
from os.path import isfile, join
from scipy.signal import savgol_filter, lfilter, hilbert
import statsmodels.api as sm

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
        
def add_meta(image, meta):
    with open(image, 'a+b') as f:
        f.write(json.dumps(meta).encode('utf-8'))

def read_meta(image):
    with open(image, 'rb') as f:
        data = str(f.read())
    meta = re.findall(r'xff.*({.*})\'\Z', data)[-1]
    return meta

again = 1

while again == 1:
    #dataset = '210718_153122_xCar_320x240_ccw'
    dataset = input("What is the name of the dataset (don't include .hdf5)\n")
    h = []
    i = 0
    foundImg = 0
    foundSteering = 0
    dict = {}
    if (os.path.isdir(dataset) == False):
        os.mkdir(dataset)
    else:
        shutil.rmtree(dataset)
        os.mkdir(dataset)

    ## -------------------------- Extract Images and Steering Data -----------------------
        
    with h5py.File(dataset + ".hdf5", 'r') as f:
        for dset in traverse_datasets(f):
            if f[dset].shape==(240, 320, 3):
                j = np.array(f[dset][:])
                array = np.reshape(j, (240, 320,3))
                im = Image.fromarray(array)
                im = ImageOps.flip(im)
                file_name = str(i)
                file_name = file_name.zfill(5)
                file_name = dataset + "/" + file_name + ".jpg"
                im.save(file_name)
                i = i+1
                foundImg = 1
            if 'steering' in dset:
                h.append(np.array(f[dset]))
                dict[i-1] = np.array(f[dset])
                foundSteering = 1
            if (foundImg == 1 and foundSteering == 1):
                add_meta(file_name , {'Steering Data' : str(np.array(f[dset]))})
                #print(read_meta(file_name))
                #print("I added metadata to " + file_name)
                foundImg = 0
                foundSteering = 0
    #Savitsky-Golay Filter
        window = 3
        order = 1
        y_sf = savgol_filter(h, window, order)

    #LOWESS smoother
        y_lowess = sm.nonparametric.lowess(h, range(i), frac = 0.02)

    #IIR filter
        n = 6             # larger n gives smoother curves
        b = [1.0 / n] * n 
        a = 1            
        y_lf = lfilter(b, a, h)

    ## ----------------------------------- Find the Landmarks -----------------------------

        values = np.array(h)
        ii = np.where(np.logical_and(values>=(max(h) - (0.015*max(h))), values<=(max(h) + (0.015*max(h)))))[0]
        #iz = np.where(np.logical_and(values>=(min(h) - (0.025*min(h))), values<=(min(h) + (0.025*min(h)))))[0]
        iz = np.where(values == min(h))[0]
        if dataset[-3:] == 'ccw':
            #print('iz')
            ii = iz
        elif dataset[-3:] == '_cw':
            #print('ii')
            ii = ii
        else:
            #print('Putting it together')
            ii = np.concatenate((ii, iz))
            
        iii = []
        iii.append(ii[0])
        n = 0
        for iv in ii:
            if n >= len(ii)-1:
                n = len(ii)-1
            else:
                n = n+1
                
            if ii[n] - iv != 1:
                iii.append(iv)
                iii.append(ii[n])
        iii.pop()

        c = input("How many images previous to the landmark?\n")

        for v in iii:
            if iii.index(v) % 2 == 0:
                iii[iii.index(v)] = v-3
            else:
                 iii[iii.index(v)] = v

        for vi in iii:
            if iii.index(vi)-1 >= 0 and iii.index(vi)-1 < len(iii):
                zzz = iii[iii.index(vi)-1]
            else:
                zzz = 0
            if vi - zzz < 5:
                iii.pop(iii.index(vi))
                iii.pop(iii.index(zzz))
                 
        for y in iii:
            plt.vlines(y, min(h), max(h), colors = 'r')
            plt.vlines(y-int(c), min(h), max(h), colors = 'g')

        iv = []
        vii = []
        iii[:] = [iii + 1 for iii in iii]
        iv[:] = [iii - int(c) for iii in iii]
        iv = iii + iv
        iv.sort()
        vii[:] = iv[:]
        iv[:] = [str(iv) for iv in iv]
        iv[:] = [iv.zfill(5) + ".jpg" for iv in iv]


    ## ----------------------------------- Moving Images into Folders -----------------------------

        folder_path = dataset
        images = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        iv.append(images[-1])
        
        counter = 0
        negpos_array = ['negative', 'positive']
        negpos = negpos_array[0]
        foldercounter = 0
        straightorno = False
        for image in images:
            if iv[counter] == image:
                counter = counter + 1
                if (counter % 2) == 0:
                    negpos = negpos_array[0]
                    if counter != len(iv):
                        foldercounter = foldercounter + 1
##                    else:
##                        counter = counter - 1
##                        if (counter % 2) == 0:
##                            negpos = negpos_array[0]
##                        else:
##                            negpos = negpos_array[1]
                elif counter == len(iv):
                    negpos = negpos_array[0]
                else:
                    negpos = negpos_array[1]
                    if straightorno == False:
                        straightorno = not straightorno
                        turndir1 = vii[counter]
                        turndir2 = vii[counter - 1]
                        turndir1 = turndir1 + 3
                        difference = h[turndir1] - h[turndir2]
                        if difference < 0:
                            negpos = "left"
                        elif difference > 0:
                            negpos = "right"
                    else:
                        negpos = "straight"
                        straightorno = not straightorno
                    

            folder_name = (str(foldercounter)).zfill(2) + "-" + negpos

            new_path = os.path.join(folder_path, folder_name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            old_image_path = os.path.join(folder_path, image)
            new_image_path = os.path.join(new_path, image)
            shutil.move(old_image_path, new_image_path)

        plt.plot(range(i),h)
        #plt.plot(range(i),y_sf)
        #plt.plot(y_lowess[:, 0], y_lowess[:, 1])
        #plt.plot(range(i), y_lf)
        plt.savefig(dataset + '/plot.png', dpi=500)
        plt.show()

        asdj = input("Run another dataset? (y/n)\n")
        if asdj == 'n':
            again = 0



