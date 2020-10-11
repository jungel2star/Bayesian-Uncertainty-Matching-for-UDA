import os
import numpy as np
import pickle
import random
from PIL import Image

dirr="D:/paper codes/PycharmProjects/data/Office31/"
dirr="D:/paper codes/PycharmProjects/data/office_caltech_10/"
datasetname="dslr"
labellist=os.listdir(dirr+datasetname+'/')
labelfile= open(dirr+'office31_label_10.txt','w')
for labeli in range(len(labellist)):
    labelfile.write(labellist[labeli]+(" %d"%labeli))
    labelfile.write("\n")
labelfile.close()


