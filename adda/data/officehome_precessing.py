import os
import numpy as np
import pickle
import random
from PIL import Image

dirr="D:/paper codes/PycharmProjects/data/OfficeHomeDataset/"
datasetname="RealWorld"   #  Clipart  Art    RealWorld  Product
labellist=os.listdir(dirr+datasetname+'/')
labelfile= open(dirr+'officeHome_label.txt','r').readlines()
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

        dataset.append(np.array(im))
        labelset.append(dict_label[labellist[labeli]])




dataset=np.array(dataset)
labelset=np.array(labelset)
print ("len(datast):",len(dataset))
print("dataset.shape:",dataset.shape,"dataset.shape:",labelset.shape)
print (dataset.shape[0])
pickle.dump((dataset,labelset),open(dirr+datasetname+"_256.pkl","wb"))
print ("end..")

