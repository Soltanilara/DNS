import numpy as np
import h5py
from PIL import Image, ImageOps
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re
import json
import os

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

def grouper(iterable):
    prev = None
    group = []
    for item in iterable:
        if not prev or item - prev <= 15:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group

h = []
i = 0
foundImg = 0
foundSteering = 0
dict = {}
if (os.path.isdir("Photos") == False):
    os.mkdir("Photos")
with h5py.File('210718_153536_xCar_320x240_ccw.hdf5', 'r') as f:
    for dset in traverse_datasets(f):
        if f[dset].shape==(240, 320, 3):
            j = np.array(f[dset][:])
            array = np.reshape(j, (240, 320,3))
            im = Image.fromarray(array)
            im = ImageOps.flip(im)
            im.save("Photos/" + str(i) + ".jpg")
            i = i+1
            foundImg = 1
        if 'steering' in dset:
            h.append(np.array(f[dset]))
            dict[i-1] = np.array(f[dset])
            foundSteering = 1
        if (foundImg == 1 and foundSteering == 1):
            add_meta("Photos/" + str(i-1) + ".jpg" , {'Steering Data' : str(np.array(f[dset]))})
            print(read_meta("Photos/" + str(i-1) + ".jpg"))
            print("I added metadata to " + str(i-1) + ".jpg")
            foundImg = 0
            foundSteering = 0
            
    g = np.array(h)
    np.savetxt("Photos/steeringdata.txt",g)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(g.reshape(-1,1))
    m = kmeans.cluster_centers_
    
    a = m[1] - m[0]
    a=a/2
    b = m[0]
    c = b+a
    g = []
    z = []
    for key in dict:
        if (c - 100)<dict[key]<(c+100): #check whether this should be a variable range
            z.append(key)
            g.append(str(key) + ':' + str(dict[key]))
    
##    z = list(grouper(z))
##    k = 0
##    for x in z:
##        j = len(x)
##        if j > 1:
##            x = x[:1] + x[j-1:]
##            z[k] = x
##        k = k+1
##
##    merged_list = []
##    for l in z:
##        merged_list += l
####    for x in merged_list:
####        plt.vlines(x,min(h),max(h), colors = 'y')
##
    values = np.array(h)
    searchval = min(h)
    ii = np.where(values == searchval)[0]
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
            print(iv)
    iii.pop()

    
    plt.plot(range(i),h)
    
    plt.vlines(41,min(h), max(h), colors = 'r')
    plt.vlines(55,min(h), max(h), colors = 'r')
    plt.vlines(56,min(h), max(h), colors = 'g')
    
    plt.vlines(106,min(h), max(h), colors = 'r')
    plt.vlines(119,min(h), max(h), colors = 'g')
    plt.vlines(120,min(h), max(h), colors = 'r')
    
    plt.vlines(255,min(h), max(h), colors = 'r')
    plt.vlines(269,min(h), max(h), colors = 'r')
    plt.vlines(270,min(h), max(h), colors = 'g')
    
    plt.vlines(352,min(h), max(h), colors = 'r')
    plt.vlines(365,min(h), max(h), colors = 'g')
    plt.vlines(366,min(h), max(h), colors = 'r')
    
    plt.vlines(405,min(h), max(h), colors = 'r')
    plt.vlines(419,min(h), max(h), colors = 'r')
    plt.vlines(420,min(h), max(h), colors = 'g')
    
    plt.vlines(494,min(h), max(h), colors = 'r')
    plt.vlines(507,min(h), max(h), colors = 'g')
    plt.vlines(508,min(h), max(h), colors = 'r')
    
    plt.vlines(628,min(h), max(h), colors = 'r')
    plt.vlines(642,min(h), max(h), colors = 'r')
    plt.vlines(643,min(h), max(h), colors = 'g')
    
    plt.vlines(721,min(h), max(h), colors = 'r')
    plt.vlines(734,min(h), max(h), colors = 'g')
    plt.vlines(735,min(h), max(h), colors = 'r')
##    plt.hlines(m[0],0,i,colors='r')
##    plt.hlines(m[1],0,i,colors='r')
##    plt.hlines(m[2],0,i,colors='r')
    plt.show()



