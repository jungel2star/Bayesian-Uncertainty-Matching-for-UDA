import os
import numpy as np
import pickle
import random
from PIL import Image
import cv2


dirr="D:/paper codes/PycharmProjects/data/Office31/"
dirr="D:/paper codes/PycharmProjects/data/office_caltech_10/"
a=0
dataset_total=["dslr","webcam","amazon","caltech"]
for datai in range(len(dataset_total)):
    datasetname=dataset_total[datai]   #  dslr  webcam    amazon   caltech
    labellist=os.listdir(dirr+datasetname+'/')
    labelfile= open(dirr+'office31_label_10.txt','r').readlines()
    dict_label={}
    dataset=[]
    labelset=[]
    for labeli in range(len(labelfile)):
        line=labelfile[labeli].split(" ")
        dict_label[str(line[0])]=int(line[1])

    for labeli in range(len(labellist)):
        imagelist=os.listdir(dirr+datasetname+'/'+labellist[labeli] +"/")
        print ("labeli:",labeli)
        for imagei in range(len(imagelist)):
            imagename=dirr+datasetname+'/'+labellist[labeli] +"/"+ imagelist[imagei]
            #print ("imagename:",imagename)
            im = Image.open(imagename).resize((256, 256),Image.ANTIALIAS)
            if not im.mode == "RGB":
                print(im.mode, imagename)
                a = a + 1
                im = im.convert('RGB')
            #im = Image.open(imagename)
            #im=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            dataset.append(np.array(im))
            labelset.append(dict_label[labellist[labeli]])



    print ("a: ", a)
    dataset=np.array(dataset)
    labelset=np.array(labelset)
    print("dataset.shape:",dataset.shape,"dataset.shape:",labelset.shape)
    pickle.dump((dataset,labelset),open(dirr+datasetname+"_10_10label.pkl","wb"))
    print ("end..")

