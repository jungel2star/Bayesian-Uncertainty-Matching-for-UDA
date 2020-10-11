import os
import numpy as np
import pickle
import random
from PIL import Image

dirr="D:/paper codes/PycharmProjects/data/image_CLEF/"
datasetname="p"

labelfile= open(dirr+"list/"+datasetname+""+'List.txt','r').readlines()
dict_label={}
dataset=[]
labelset=[]
a=0
savedirr=dirr+datasetname+"_save/"
for labeli in range(len(labelfile)):
    imagename=labelfile[labeli].split(" ")[0].split("/")[-1]
    labelname = labelfile[labeli].split(" ")[-1]
    savename=savedirr+imagename
    imagename =dirr+datasetname+'/'+imagename
    #print (imagename)  gray=img.convert('L')

    im = Image.open(imagename).resize((224, 224), Image.ANTIALIAS)

    if not im.mode =="RGB":
        print (im.mode,imagename)
        a=a+1
        im=im.convert('RGB')
        im.save(savename)
    dataset.append(np.array(im))
    labelset.append(int(labelname))

print ("a",a)
dataset=np.array(dataset)
labelset=np.array(labelset)
print ("len(datast):",len(dataset))
print("dataset.shape:",dataset.shape,"dataset.shape:",labelset.shape)
print (dataset.shape[0])
pickle.dump((dataset,labelset),open(dirr+datasetname+".pkl","wb"))
print ("end..")

