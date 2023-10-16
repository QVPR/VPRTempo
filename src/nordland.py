'''
Imports
'''
import os
import re
import shutil
import zipfile
import sys
sys.path.append('..//dataset')

from os import walk

def nord_sort():
    # load and sort the file names in order, not by how OS indexes them
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
    # set the base path to the location of the downloaded Nordland datasets
    basePath = '../dataset/'
    assert(os.path.isdir(basePath)),"Please set the basePath to the location of the downloaded Nordland datasets"
    
    # define the subfolders of the Nordland datasets
    subPath = ["spring_images_train/section1/","spring_images_train/section2/",
               "fall_images_train/section1/","fall_images_train/section2/",
               "winter_images_train/section1/","winter_images_train/section2/",
               "summer_images_train/section1/","summer_images_train/section2/"]
    
    # set the desired output folder for unzipping and organization
    outDir = '../dataset/'
    assert(os.path.isdir(outDir)),"Please set the outDir to the desired output location for unzipping the Nordland datasets"
    
    # define output paths for the data
    outPath = [os.path.join(outDir,"spring/"),os.path.join(outDir,"fall/"),
               os.path.join(outDir,"winter/"),os.path.join(outDir,"summer/")]
    
    # check for existence of the zip folders, throw exception if missing
    zipNames = ["spring_images_train.zip","fall_images_train.zip",
                "winter_images_train.zip","summer_images_train.zip"]
    for n in zipNames:
        if not os.path.exists(basePath+n):
            raise Exception('Please ensure dataset .zip folders have been downloaded') 
    
    # check if nordland data folders have already been unzipped
    zip_flag = []
    for n, ndx in enumerate(range(0,len(subPath),2)):
        print('Unzipping '+zipNames[n])
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
    
    # if output folders already exist, delete them
    for n in outPath:
        if os.path.exists(n):
            shutil.rmtree(n)
            print('Removed pre-existing output folder')
        
    # rename and move the training data to match the nordland_imageNames.txt file
    for n in outPath:
        os.mkdir(n)
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
            out = outPath[0]
        elif "fall" in filename:
            out = outPath[1]
        elif "winter" in filename:
            out = outPath[2]
        else:
            out = outPath[-1]
    
        fileDest = out + my_dest   
        os.rename(filename, fileDest)
    
    # remove the empty folders
    for n, ndx in enumerate(subPath):
        if n%2 == 0:
            shutil.rmtree(basePath+ndx.replace('section1/',''))
        else:
            continue
    
    print('Finished unzipping and organizing Nordland dataset')