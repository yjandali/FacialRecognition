import os, sys
import random
import shutil

#CHANGE THESE TWO VARIABLES TO PROPER PATHS
ROOTDIR="~/Desktop/Projects/FacialRecog/FacialRecognition/images"
DEST = "~/Desktop/Projects/FacialRecog/FacialRecognition/Pictures"
count = 1
dirArray = []
dests = ["Train", "Validate", "Test"]

k=0
for dir in dests:
    dests[k] = DEST + "/" + dir + "/"
    os.mkdir(dests[k])
    k+=1

for subdir in os.walk(ROOTDIR):
    if count == 1:
        dirArray = subdir[1]
    count+= 1

print(ROOTDIR + "/" + dirArray[0])

for dir in dirArray:
    os.mkdir(dests[0] + dir)
    os.mkdir(dests[1] + dir)
    os.mkdir(dests[2] + dir)
    for i, j, k in os.walk(ROOTDIR + "/" + dir):
        random.shuffle(k)
        count2 = 0
        for fileName in k:
#CHANGE THE MODDED VALUES TO GET DIFFERENT PROPORTIONS
            if (count2 % 10 <= 6):
                shutil.copy2(ROOTDIR + "/" + dir + "/" + fileName, dests[0] + dir)
            elif (count2 % 10 <= 8):
                shutil.copy2(ROOTDIR + "/" + dir + "/" + fileName, dests[1] + dir)
            elif (count2 % 10 <= 9):
                shutil.copy2(ROOTDIR + "/" + dir + "/" + fileName, dests[2] + dir)
            count2+=1

