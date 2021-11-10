import numpy as np
import math
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
import plotly.express as px

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
        
def percentage_diff(num1, num2):
    a = abs(num1-num2)
    b = (num1+num2) / 2
    c = (a/b) * 100
    return c

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
                foundImg = 0
                foundSteering = 0

    ## ----------------------------------- Find the Landmarks -----------------------------
##        file = open(dataset + ".txt", "w+")
##        file.write(str(h))
##        file.close()

        #file = open("210718_153122_xCar_320x240_ccw.txt", "r")
        
##        h = file.read()
        

        
        values = np.array(h)
        ii = np.where(np.logical_and(values>=(max(h) - (0.015*max(h))), values<=(max(h) + (0.015*max(h)))))[0]
        #iz = np.where(np.logical_and(values>=(min(h) - (0.025*min(h))), values<=(min(h) + (0.025*min(h)))))[0]
        iz = np.where(values == min(h))[0]


        if ((max(h) - h[0]) - (h[0] - min(h))) > 0:
            if ((max(h) - h[0]) - (h[0] - min(h))) < 50:
                ii = np.concatenate((ii,iz))
            else:
                ii = ii
        elif ((max(h) - h[0]) - (h[0] - min(h))) < 0:
            if abs(((max(h) - h[0]) - (h[0] - min(h)))) < 50:
                ii = np.concatentate((ii,iz))
            else:
                ii = iz

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

    ## ----------------------------------- Removing False Positives -----------------------------

        falsepos = False
        zz = []
        defaults = input("all defaults? (y/n)\n")
        if defaults == 'y':
            window = 5
            floor = 100*math.floor(h[0]/100)
            ceil = 100*math.ceil(h[0]/100)
            tooclose = 10
        else:
            window = input("How many frames around landmarks should the window be (d for default)(odd number)\n")
            if window == 'd':
                window = 5
            floor = input("The current floor is " + str(100*math.floor(h[0]/100)) + ". press d to continue with this, otherwise input floor\n")
            if floor == 'd':
                floor = 100*math.floor(h[0]/100)
            ceil = input("The current ceiling is " + str(100*math.ceil(h[0]/100)) + ". press d to continue with this, otherwise input ceiling\n")
            if ceil == 'd':
                ceil = 100*math.ceil(h[0]/100)
            tooclose = input("two landmarks cannot be closer than x frames. x=? (d for default)\n")
            if tooclose == 'd':
                tooclose = 10

                
        for vi in iii:
            falsepos = False
            if math.isclose(percentage_diff(h[vi], h[vi+1]),percentage_diff(h[vi], h[vi+2]), abs_tol = 1):
                if percentage_diff(h[vi], h[vi+10]) - percentage_diff(h[vi], h[vi+1]) > 1.5:
                    if percentage_diff(h[vi], h[vi+15]) > 10:
                        falsepos = False
                    elif percentage_diff(h[vi], h[vi+20]) > 20:
                        falsepos = False
                    else:
                        falsepos = True
                else:
                    falsepos = True
            if falsepos == True:
                zz.append(iii.index(vi))                
        
        for vii in sorted(zz, reverse=True):
            del iii[vii]

        roc = []
        roc = iii.copy()
        
        for aaa in range(len(roc)):
            for bbb in range(1,int(window)+1):
                roc.append(roc[aaa] - bbb)
                roc.append(roc[aaa] + bbb)

        roc.sort()
        #print(roc)
        #print(iii)
        
        indexofroc = 0
        countldmrks = 0
        while(indexofroc!=len(roc)):
            landmark = iii[countldmrks]
            #print(landmark)
            rocarray = []
            rocarrayframes = []
            changeldmrk = 0
            newldmrk = 0
            change = False

            for elem in range(0,(2*int(window))+1):
                if elem != 2*(int(window))+1:
                    rocarrayframes.append(roc[elem + indexofroc])
            #print(h[0])
            isCenter = False
            if countldmrks %2 == 0:
                while(isCenter == False):
                    isCenter = False
                    #print("ran")
                    #print("Ceil = " + str(ceil))
                    #print("Floor = " + str(floor))
                    #print(rocarrayframes)
                    for frames in rocarrayframes:
                        #print(h[frames])
                        if (int(ceil) > h[frames] > int(floor)):
                            #print("I went in")
                            isCenter = True
                    if isCenter == False:
                        for frame in range(0,len(rocarrayframes)):
                            #print(rocarrayframes[frame])
                            rocarrayframes[frame] = rocarrayframes[frame] - int(window)
                            #print(rocarrayframes[frame])
                            
            for elem in range(0,len(rocarrayframes)):
                if elem != len(rocarrayframes)-1:
                    rocarray.append(abs((h[rocarrayframes[elem+1]] - h[rocarrayframes[elem]]) / 2.0))
        
            #print(rocarrayframes)
            #print(rocarray)
            #print(iii)
            
            maxindex = rocarrayframes[rocarray.index(max(rocarray))] + 1
            #print("Maxindex = " + str(maxindex))
            if rocarray.index(max(rocarray))+1 == 1:
                rocarray = rocarray[::-1]
                rocarrayframes = rocarrayframes[::-1]
                maxindex = maxindex + 2
            fixldmrk = range(0,rocarray.index(max(rocarray))+1)
            fixldmrk = fixldmrk[::-1]
            for elements in fixldmrk:
                if elements + 1 < rocarray.index(max(rocarray))+1:
                    #print(rocarrayframes[elements], rocarrayframes[elements + 1], rocarray[elements], rocarray[elements + 1])
                    if rocarray[elements] < rocarray[elements + 1] or rocarray[elements+1] >= 10:
                        changeldmrk = changeldmrk + 1
                        #print(changeldmrk)
                        if elements - 1 >= 0:
                            if rocarray[elements] < 10 and rocarray[elements - 1] > 10:
                                changeldmrk = changeldmrk + 1
                            if rocarray[elements] < 10 and rocarray[elements - 1] < 10:
                                #print("end here")
                                #print(maxindex-changeldmrk)
                                iii[countldmrks] = maxindex-changeldmrk
                                change = True
                                break
                        else:
                            if rocarray[elements] < 10:
                                #print("end here")
                                #print(maxindex-changeldmrk)
                                iii[countldmrks] = maxindex-changeldmrk
                                change = True
                                break

            if changeldmrk > 0 and change == False:
                iii[countldmrks] = maxindex - changeldmrk
                print("Warning: Landmark at frame " + str(iii[countldmrks]) + " is inaccurate, increase window!")
                            
            
            indexofroc = indexofroc+(2*(int(window)))+1
            countldmrks = countldmrks + 1
            #print("repeat")
        
        for ldmarks in iii:
            if iii.index(ldmarks)-1 >= 0 and iii.index(ldmarks)-1 < len(iii):
                zzz = iii[iii.index(ldmarks)-1]
            else:
                zzz = 0
            if ldmarks - zzz < tooclose:
                #print(ldmarks, zzz)
                iii.pop(iii.index(ldmarks))
                iii.pop(iii.index(zzz))
                
        figure = px.line(x=range(i), y = h)
        for y in iii:
            figure.add_vline(x = y, line_color = 'red')
            figure.add_vline(x = y - int(c), line_color = 'green')
        figure.show()

        print(iii)
        change = 0
        while (change != 'no'):
            change = input("Do you want to change/delete landmarks or add landmarks? (cd/a/no)\n")
            if change == 'no':
                break
            elif change == 'cd':
                for yy in iii:
                    zoomed = px.line(x = range(yy-int(c), yy+int(c)+1), y = [h[find] for find in range(yy-int(c), yy+int(c)+1)])
                    zoomed.add_vline(x = yy, line_color = 'red')
                    zoomed.update_xaxes(nticks=2*int(c))
                    zoomed.show()
                    action = input("What action would you like to take about this landmark? (n = N/A, d = del, c = change, exit)\n")
                    if action == 'n':
                        print("No change")
                    elif action == 'd':
                        iii.pop(iii.index(yy))
                        print(iii)
                    elif action == 'c':
                        iii[iii.index(yy)] = input("What is the correct landmark frame?\n")
                        print(iii)
                    elif action == 'exit':
                        break
            elif change == 'a':
                addldmrk = input("What landmark would you like to add?\n")
                zoomed = px.line(x = range(addldmrk-int(c), addldmrk+int(c)+1), y = [h[find] for find in range(addldmrk-int(c), addldmrk+int(c)+1)])
                zoomed.add_vline(x = addldmrk, line_color = 'red')
                zoomed.update_xaxes(nticks=2*int(c))
                zoomed.show()
                addldmrk = input("What landmark would you like to add? (input frame of landmark again to confirm)\n")
                iii.append(int(addldmrk))
                iii.sort()
                print("Added landmark at " + str(addldmrk) + "\n")


        finalfigure = px.line(x=range(i), y = h)
        for y in iii:
            finalfigure.add_vline(x = y, line_color = 'red')
            finalfigure.add_vline(x = y - int(c), line_color = 'green')

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

        p = os.listdir(dataset)
        if len(p)%2 != 0:
            src = dataset + '/' + p[len(p)-1] + '/'
            dest = dataset + '/00-negative/'
            allfiles = os.listdir(src)
            for negimage in allfiles:
                shutil.move(src+negimage, dest+negimage)
            os.rmdir(src)
            


        finalfigure.update_layout(title_text="Final Graph")
        finalfigure.write_html(dataset + '/plot.html')
        finalfigure.show()

        

        asdj = input("Run another dataset? (y/n)\n")
        if asdj == 'n':
            again = 0



