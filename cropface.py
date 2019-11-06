# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:42:03 2019

@author: zjj
"""

import dlib
import cv2
import numpy as np
import glob
'''
crop the face for the whole dataset
'''
def cropface_dataset(input_folder,output_folder):
    detector = dlib.get_frontal_face_detector()
    
                   
    count = 0
    for filename in glob.glob(input_folder+'/*'):
        count += 1
        if count == 45000:
            break
        try: # to avoid potential error when cropping face
            img = cv2.imread(filename)
            dets = detector(img)
            print(filename)
            if len(dets) == 0:
                cv2.imwrite(output_folder+'/'+filename[-10:], img)
                pass
            else:
                for k, d in enumerate(dets):
                  # read the coordinates of the cropped face
                  # calculate the size of the face
                  height = d.bottom()-d.top()
                  width = d.right()-d.left()
                  # generate empty image for cropped face
                  img_blank = np.zeros((height, width, 3), np.uint8)
                  # put the face into empty image
                  for i in range(height):
                    for j in range(width):
                      img_blank[i][j] = img[d.top()+i][d.left()+j]
                  cv2.imwrite(output_folder+'/'+filename[-10:], img_blank)
        except IndexError:
            cv2.imwrite(output_folder+'/'+filename[-10:], img)
            pass
        continue   
    
#cropface_dataset('data','cropped')

#%%
''' crop multi face in a single images
    only crop test image,one image per time
'''
              
def cropface_own(path):
#    path = 'testfinal.jpg'
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(path)
    dets = detector(img)
    print(len(dets))
    
    for k, d in enumerate(dets):
      # read the coordinates of the cropped face
      # calculate the size of the face
      height = d.bottom()-d.top()
      width = d.right()-d.left()
      # generate empty image for cropped face
      img_blank = np.zeros((height, width, 3), np.uint8)
      # put the face into empty image
      for i in range(height):
        for j in range(width):
          img_blank[i][j] = img[d.top()+i][d.left()+j]
      cv2.imwrite(str(k)+path, img_blank)
      cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255,0,0), 2)
#    cv2.imwrite('multiface.jpg', img)
    cv2.imshow("lalala", img)
    
    k = cv2.waitKey(0) 
#cropface_own('multitest.jpg')
