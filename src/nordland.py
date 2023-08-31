#MIT License

#Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''
import os
import re
import shutil
import zipfile

from os import walk

# load and sort the file names in order, not by how OS indexes them
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

basePath = "/home/adam/data/VPRTempo_training/"
subPath = ["spring_images_train/section1/","spring_images_train/section2/",
           "fall_images_train/section1/","fall_images_train/section2/",
           "winter_images_train/section1/","winter_images_train/section2/",
           "summer_images_train/section1/","summer_images_train/section2/"]
outPath = ["training_data/spring/","training_data/fall/","training_data/winter/","testing_data/"]

# check for existence of the zip folders, throw exception if missing
zipNames = ["spring_images_train.zip","fall_images_train.zip",
            "winter_images_train.zip","summer_images_train.zip"]
for n in zipNames:
    if not os.path.exists(basePath+n):
        raise Exception('Please ensure dataset .zip folders have been downloaded') 

# check if nordland data folders have already been unzipped
zip_flag = []
for n, ndx in enumerate(range(0,len(subPath),2)):
    if os.path.exists(basePath+subPath[ndx]):
        # check if the folder contains any files
        file_lst = os.listdir(basePath+subPath[ndx])
        # remove folder if it is empty and unzip the data folder
        if len(file_lst) == 0:
            shutil.rmtree(basePath+subPath[ndx].replace('section1/',''))
            with zipfile.ZipFile(basePath+zipNames[n],"r") as zip_ref:
                zip_ref.extractall(basePath)
    else:
        with zipfile.ZipFile(basePath+zipNames[n],"r") as zip_ref:
            zip_ref.extractall(basePath)
            
# load image paths
tempPaths = []
imgPaths = []
for n in range(0,len(subPath)):
    tempPaths = []
    for (path, dir_names, file_names) in walk(basePath+subPath[n]):
        tempPaths.extend(file_names)   
    # sort image names
    tempPaths.sort(key=natural_keys)
    tempPaths = [basePath+subPath[n]+s for s in tempPaths]
    imgPaths = imgPaths + tempPaths

# if training and testing folders already exist, delete them
if os.path.exists(basePath+'training_data/'):
    shutil.rmtree(basePath+'training_data/')
    print('Removed pre-existing training data folder')
if os.path.exists(basePath+'testing_data/'):
    shutil.rmtree(basePath+'testing_data/')
    print('Remove pre-existing testing data folder')
    
# rename and move the training data
os.mkdir(basePath+"training_data/")
for n in outPath:
    os.mkdir(basePath+n)
for n, filename in enumerate(imgPaths):
    base = os.path.basename(filename)
    split_base = os.path.splitext(base)
    if int(split_base[0]) < 10:
        my_dest = "images-0000"+split_base[0] + ".png"
    elif int(split_base[0]) < 100:
        my_dest = "images-000"+split_base[0] + ".png"
    elif int(split_base[0]) < 1000:
        my_dest = "images-00"+split_base[0] + ".png"
    elif int(split_base[0]) < 10000:
        my_dest = "images-0"+split_base[0] + ".png"
    else:
        my_dest = "images-"+split_base[0] + ".png"
    if "spring" in filename:
        out = basePath + outPath[0]
    elif "fall" in filename:
        out = basePath + outPath[1]
    elif "winter" in filename:
        out = basePath + outPath[2]
    else:
        out = basePath + outPath[-1]

    fileDest = out + my_dest   
    os.rename(filename, fileDest)
    
