# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:24:05 2017

@author: lab
"""

import xlrd, os, urllib, cStringIO
import numpy as np
import pylab as plt
from PIL import Image


# Define path to git repo
main_dir=os.environ['HOME']+"/Cloud2/movies/human/turk/face_sim/"

FNAMES = ['ArnoldBarney','BarneyDaniel','DanielHillary','DanielShinzo','HillaryShinzo','IanPiers','IanTom','PiersTom'];      
FPAIR1 = [0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6];
FPAIR2 = [1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7];

FACE_URL = ['http://i.imgur.com/HFMr5Jp.png','http://i.imgur.com/VoB4Hr1.png','http://i.imgur.com/8afrTze.png','http://i.imgur.com/vvvahSz.png','http://i.imgur.com/xZLTaKf.png','http://i.imgur.com/M1g57XX.png','http://i.imgur.com/XA8A3FR.png','http://i.imgur.com/Jbh4AeB.png'];

# import data
workbook = xlrd.open_workbook(main_dir+'results/test_batch.xlsx');
sheet = workbook.sheet_by_index(0)

# Allocate imported worksheet to column variable names
Answerface1 = sheet.col_values(0)
Answerface2 = sheet.col_values(1)
AnswerfacePairNum = sheet.col_values(2)
Answerresponses = sheet.col_values(3)
AnswertrialNum = sheet.col_values(4)

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

fig, ax = plt.subplots(1, figsize=(5,30), facecolor='white')

plt.subplot(111)
plt.imshow(cat_resp,interpolation='none',cmap='hot')
plt.xlabel('face-pair')
plt.ylabel('HIT')
plt.title('normalized responses')
plt.colorbar()
#plt.show()

fig.savefig(main_dir+'results/norm_resps.png', facecolor=fig.get_facecolor(), edgecolor='none')

# Calc mean response
mean_resp = cat_resp.mean(axis=0)

# Pre-allocate similarity matrix
resp_mat = np.ones(shape=(len(FNAMES),len(FNAMES)))*min(mean_resp)

# Loop through face-pairs
for i in range(FPAIR1[-1]+1):
   
    # Search out indices of each instance of current pair
    pair_indices = [j for j,x in enumerate(FPAIR1) if x==i]

    for ii,x in enumerate(pair_indices):
        
            resp_mat[ii,i] = mean_resp[x]
#            print((ii,i))

fig, ax = plt.subplots(1, figsize=(10,10), facecolor='black')

plt.subplot(111)
plt.imshow(resp_mat,interpolation='none',cmap='hot') # ,aspect=1
#plt.colorbar(orientation='horizontal')

xl, yl, xh, yh=np.array(ax.get_position()).ravel()
w = xh-xl
h = yh-yl
size = w/len(FNAMES)

crop_size = 20

for i,x in enumerate(FACE_URL):
    
    xp = xl+(size*i)
    yp = yl+(size*i)

    # Load image using URL
    file = cStringIO.StringIO(urllib.urlopen(x).read())
    face_img = Image.open(file)

    # Crop image edges
    face_img = face_img.crop((crop_size,crop_size,face_img.size[0]-crop_size,face_img.size[1]-crop_size))

    # Plot on x-axis
    ax1=fig.add_axes([xp, yh-0.01, size, size])
    ax1.axison = False
    imgplot = ax1.imshow(face_img)

    # Plot on y-axis
    ax2=fig.add_axes([xl-size-0.01, yp+0.02, size, size])
    ax2.axison = False
    imgplot = ax2.imshow(face_img)

# Create colorbar content
foo = np.array([range(100) for x in range(3)])

# Plot colorbar manually
ax1 = fig.add_axes([xl, 0.05, w, h/13.5])
im = ax1.imshow(foo,interpolation='nearest',cmap='hot')
ax1.set_xlabel('mean similarity',color='white')
ax1.set_xticks((0,100))
ax1.set_yticks([])
temp_lims = (round(min(mean_resp)*1000)/1000,round(max(mean_resp)*1000)/1000)
ax1.set_xticklabels(temp_lims,color='white')

plt.show()

fig.savefig(main_dir+'results/face_dsm.png',dpi=200,facecolor=fig.get_facecolor(), edgecolor='none')

pass



