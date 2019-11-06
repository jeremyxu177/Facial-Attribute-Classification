# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:26:20 2019

@author: zjj
"""


import torch
import numpy as np
import os
from torch.autograd import Variable
import linecache as lc
from dataloader import *
from model import *
from skimage import io
import cv2
from PIL import Image, ImageDraw, ImageFont
from cropface import cropface_own

torch.cuda.set_device(0)
cwd = os.getcwd()
print(cwd)
model = AttrPre()
model.cuda()
#  open the saved model 
checkpoint = torch.load('M:/Data_Baidu/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png.7z/params.pkl', map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model_state_dict'])

### select the index of test images.
lineNum = 10000
it=iter(range(50000,60000))
# calculate total number of correct labelled features
AttractiveCounter = 0
EyeGlassesCounter = 0
MaleCounter = 0
MouthOpenCounter = 0
SmilingCounter = 0
YoungCounter = 0
browncounter=0
hatcounter=0
lipcounter=0
ovalcounter = 0
'''
performs model on the test set.
compare the results and record the correct number
'''
label_=os.listdir('data')
for m in it:
    line = lc.getline('list_attr_celeba.txt', m+2)
    line = line.rstrip('\n')
    file = line.split()

    ImgName = os.path.join('data/',
                              label_[m])
    iAttractive = []
    iEyeGlasses = []
    iMale = []
    iMouthOpen = []
    iSmiling = []
    iYoung = []
    ibrownhair = []
    ihat = []
    ilipstick = []
    iovalface = []
    iAttractive.append(float(file[3]))
    iEyeGlasses.append(float(file[16]))
    iMale.append(float(file[21]))
    iMouthOpen.append(float(file[22]))
    iSmiling.append(float(file[32]))
    iYoung.append(float(file[40]))
    ibrownhair.append(float(file[23])) 
    ihat.append(float(file[36])) 
    ilipstick.append(float(file[37])) 
    iovalface.append(float(file[26])) 
    iAttractive = np.asarray(iAttractive)
    iEyeGlasses = np.asarray(iEyeGlasses)
    iMale = np.asarray(iMale)
    iMouthOpen = np.asarray(iMouthOpen)
    iSmiling = np.asarray(iSmiling)
    iYoung = np.asarray(iYoung)
    ibrownhair= np.asarray(ibrownhair)
    ihat= np.asarray(ihat)
    ilipstick= np.asarray(ilipstick)
    iovalface= np.asarray(iovalface) 
    input = io.imread(ImgName)
    if input.ndim < 3:
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(input, (178,218))
    imgI = (torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)-0.5)/0.5
    imgI = imgI.cuda()
    imgI = Variable(imgI)
    model.eval()
    AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre,brownPre,hatPre,lipPre,ovalPre = model(imgI)
    AttractiveP = AttractivePre.cpu().data.numpy()[0]
    EyeGlassesP = EyeGlassesPre.cpu().data.numpy()[0]
    MaleP = MalePre.cpu().data.numpy()[0]
    MouthOpenP = MouthOpenPre.cpu().data.numpy()[0]
    SmilingP = SmilingPre.cpu().data.numpy()[0]
    YoungP = YoungPre.cpu().data.numpy()[0]
    brownP = brownPre.cpu().data.numpy()[0]
    hatP = hatPre.cpu().data.numpy()[0]
    lipP = lipPre.cpu().data.numpy()[0]
    ovalP = ovalPre.cpu().data.numpy()[0]
    # if mark smaller than 0.5, class 0. else class 1
    if AttractiveP <0.5:
        if iAttractive[0] == -1:
            AttractiveCounter = AttractiveCounter +1
    else:
        if iAttractive[0] == 1:
            AttractiveCounter = AttractiveCounter +1
    if EyeGlassesP <0.5:
        if iEyeGlasses[0] == -1:
            EyeGlassesCounter = EyeGlassesCounter +1
    else:
        if iEyeGlasses[0] == 1:
            EyeGlassesCounter = EyeGlassesCounter +1
    if MaleP <0.5:
        if iMale[0] == -1:
            MaleCounter = MaleCounter +1
    else:
        if iMale[0] == 1:
            MaleCounter = MaleCounter +1
    if MouthOpenP <0.5:
        if iMouthOpen[0] == -1:
            MouthOpenCounter = MouthOpenCounter +1
    else:
        if iMouthOpen[0] == 1:
            MouthOpenCounter = MouthOpenCounter +1
    if SmilingP <0.5:
        if iSmiling[0] == -1:
            SmilingCounter = SmilingCounter +1
    else:
        if iSmiling[0] == 1:
            SmilingCounter = SmilingCounter +1
    if YoungP <0.5:
        if iYoung[0] == -1:
            YoungCounter = YoungCounter +1
    else:
        if iYoung[0] == 1:
            YoungCounter = YoungCounter +1
    if brownP <0.5:
        if ibrownhair[0] == -1:
            browncounter = browncounter +1
    else:
        if ibrownhair[0] == 1:
            browncounter = browncounter +1
    if hatP <0.5:
        if ihat[0] == -1:
            hatcounter = hatcounter +1
    else:
        if ihat[0] == 1:
            hatcounter = hatcounter +1       
    if lipP <0.5:
        if ilipstick[0] == -1:
            lipcounter = lipcounter +1
    else:
        if ilipstick[0] == 1:
            lipcounter = lipcounter +1
    if ovalP <0.5:
        if iovalface[0] == -1:
            ovalcounter = ovalcounter +1
    else:
        if iovalface[0] == 1:
            ovalcounter = ovalcounter +1
    print(m)
    # print the results
print('attractive: '+str(AttractiveCounter/lineNum))
print('wear eye glasses: '+str(EyeGlassesCounter/lineNum))
print('male: '+str(MaleCounter/lineNum))
print('mouth slightly open: '+str(MouthOpenCounter/lineNum))
print('smiling: '+str(SmilingCounter/lineNum))
print('young: '+str(YoungCounter/lineNum))
print(' moustache: '+str(browncounter/lineNum))
print('wear hat: '+str(hatcounter/lineNum))
print('wear lipstick: '+str(lipcounter/lineNum))
print('oval face: '+str(ovalcounter/lineNum))



#%% test on our own photos
filename = 'test/123456.jpg'
input = io.imread(filename)

if input.ndim < 3:
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
inp = cv2.resize(input, (178,218))
imgI = (torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)-0.5)/0.5
imgI = imgI.cuda()
imgI = Variable(imgI)
model.eval()
AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre,brownPre,hatPre,lipPre,ovalPre = model(imgI)
AttractiveP = AttractivePre.cpu().data.numpy()[0]
EyeGlassesP = EyeGlassesPre.cpu().data.numpy()[0]
MaleP = MalePre.cpu().data.numpy()[0]
MouthOpenP = MouthOpenPre.cpu().data.numpy()[0]
SmilingP = SmilingPre.cpu().data.numpy()[0]
YoungP = YoungPre.cpu().data.numpy()[0]
brownP = brownPre.cpu().data.numpy()[0]
hatP = hatPre.cpu().data.numpy()[0]
lipP = lipPre.cpu().data.numpy()[0]
ovalP = ovalPre.cpu().data.numpy()[0]

print('attractive: '+str(AttractiveP))
print('wear eye glasses: '+str(EyeGlassesP))
print('male: '+str(MaleP))
print('mouth slightly open: '+str(MouthOpenP))
print('smiling: '+str(SmilingP))
print('young: '+str(YoungP))
print(' moustache: '+str(brownP))
print('wear hat: '+str(hatP))
print('wear lipstick: '+str(lipP))
print('oval face: '+str(ovalP))

if AttractiveP <0.5:
    AttractiveP = 0  
else:
    AttractiveP = 1
if EyeGlassesP <0.5:
    EyeGlassesP = 0
else:
    EyeGlassesP = 1
if MaleP <0.5:
    MaleP = 0
else:
    MaleP = 1
if MouthOpenP <0.5:
    MouthOpenP = 0
else:
    MouthOpenP = 1
if SmilingP <0.5:
    SmilingP = 0
else:
    SmilingP = 1
if YoungP <0.5:
    YoungP = 0
else:
    YoungP = 1
if brownP <0.5:
    brownP = 0
else:
    brownP = 1
if hatP <0.5:
    hatP = 0
else:
    hatP = 1
        
if lipP <0.5:
    lipP = 0
else:
    lipP = 1
if ovalP <0.5:
    ovalP = 0
else:
    ovalP = 1
image = Image.open(filename)
# initialise the drawing context with the image object as background
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 70)
# starting position of the message
(x, y) = (50, 10)
message = 'attractive: '+str(AttractiveP)+'\n'+'wear eye glasses: '+str(EyeGlassesP) \
+'\n'+'male: '+str(MaleP)+'\n'+'mouth slightly open: '+str(MouthOpenP)+'\n'+'smiling: '+str(SmilingP)+'\n'+'moustache: '+str(brownP)\
+'\n'+'wear hat: '+str(hatP)+'\n'+'wear lipstick: '+str(lipP)+'\n'+'oval face: '+str(ovalP)
color = 'rgb(0, 0, 0)' # black color
draw.text((x, y), message, fill=color,font=font)
image.show()
#image.save('finaltest3.jpg')

 
