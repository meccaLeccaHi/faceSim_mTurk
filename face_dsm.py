# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:24:05 2017

@author: lab
"""

import xlrd
import numpy as np
import pylab as plt
import urllib, cStringIO
from PIL import Image



FNAMES = ['ArnoldBarney','BarneyDaniel','DanielHillary','DanielShinzo','HillaryShinzo','IanPiers','IanTom','PiersTom'];      
FPAIR1 = [0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6];
FPAIR2 = [1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7];

FACE_URL = ['http://i.imgur.com/HFMr5Jp.png','http://i.imgur.com/VoB4Hr1.png','http://i.imgur.com/8afrTze.png','http://i.imgur.com/vvvahSz.png','http://i.imgur.com/xZLTaKf.png','http://i.imgur.com/M1g57XX.png','http://i.imgur.com/XA8A3FR.png','http://i.imgur.com/Jbh4AeB.png'];

# import data
workbook = xlrd.open_workbook('/home/lab/Cloud2/movies/human/turk/face_sim/results/test_batch.xlsx');
sheet = workbook.sheet_by_index(0)

# Allocate imported worksheet to column variable names
Answerface1 = sheet.col_values(0)
Answerface2 = sheet.col_values(1)
AnswerfacePairNum = sheet.col_values(2)
Answerresponses = sheet.col_values(3)
AnswertrialNum = sheet.col_values(4)

# Pre-allocate similarity matrix
resp_mat = np.zeros(shape=(len(FNAMES),len(FNAMES)))

def normResp(hit_num):
    """De-randomize, normalize, and concatenate responses for each HIT"""
    
    # Convert strings to ints
    responses = np.array(map(int, Answerresponses[hit_num].split(',')))
    order = np.array(map(int, AnswerfacePairNum[hit_num].split(',')))

    # De-randomize
    derand_resp = responses[order]

    # Normalize b/t 0 and 1
    norm_resp = derand_resp - derand_resp.min()    
    norm_resp = norm_resp/float(norm_resp.max())
    
    return norm_resp
        
# Loop through HITS, concatenate response vector from each
cat_resp = np.array([normResp(n) for n in range(1,len(Answerface1))])

fig, ax = plt.subplots(1, figsize=(10,10), facecolor='black')


#fig = plt.figure()
#ax = plt.subplot(111)

##im = plt.imshow(mean_resp,interpolation='bilinear',origin='lower',extent=(-3,3,-3,3))
#im = plt.imshow(cat_resp,interpolation='none',cmap='hot')
#plt.colorbar(im, orientation='horizontal')

# Calc mean response
mean_resp = cat_resp.mean(axis=0)

# Loop through face-pairs
for i in range(FPAIR1[-1]+1):
   
    # Search out indices of each instance of current pair
    pair_indices = [j for j,x in enumerate(FPAIR1) if x==i]

    for ii,x in enumerate(pair_indices):
        
            resp_mat[ii,i] = mean_resp[x]
#            print((ii,i))

#ax = plt.subplot(212)

axS = ax.matshow(resp_mat,cmap='hot',extent=[1,len(FNAMES)+1,1,len(FNAMES)+1],aspect=1)
ax.axison = False

xl, yl, xh, yh=np.array(ax.get_position()).ravel()
w = xh-xl
h = yh-yl
size = w/len(FNAMES)

crop_size = 20

for i,x in enumerate(FACE_URL):
    
    xp = xl+(size*i)
    
    # Load image using URL
    file = cStringIO.StringIO(urllib.urlopen(x).read())
    face_img = Image.open(file)

    # Crop image edges
    face_img = face_img.crop((crop_size,crop_size,face_img.size[0]-crop_size,face_img.size[1]-crop_size))

    ax1=fig.add_axes([xp, yh-0.01, size, size])
    ax1.axison = False
    imgplot = ax1.imshow(face_img)

    yp = yl+(size*i)

    ax2=fig.add_axes([xl-size-0.01, yp+0.02, size, size])
    ax2.axison = False
    imgplot = ax2.imshow(face_img)


plt.show()

# fig.savefig('whatever.png', facecolor=fig.get_facecolor(), edgecolor='none')

pass



