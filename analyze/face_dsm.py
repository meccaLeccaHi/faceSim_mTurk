# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:24:05 2017

@author: lab
"""

import xlrd, os, urllib, cStringIO
import numpy as np
import pylab as plt
from PIL import Image
from scipy.stats import rankdata
from cmdscale import cmdscale

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
    
def rankResp(hit_num):
    """De-randomize, rank-order, and concatenate responses for each HIT"""
    
    # Convert strings to ints
    responses = np.array(map(int, Answerresponses[hit_num].split(',')))
    order = np.array(map(int, AnswerfacePairNum[hit_num].split(',')))

    # De-randomize
    derand_resp = responses[order]

    # Rank-order responses
    rank_resp = rankdata(derand_resp)
#    rank_resp = norm_resp/float(norm_resp.max())
    
    return rank_resp

def plot_resps(cat_resp,main_dir,label,savename):
    """ Plot responses """
    
    fig, ax = plt.subplots(1,figsize=(5,11),facecolor='white')
    
    plt.subplot(111)
    plt.imshow(cat_resp,interpolation='none',cmap='hot')
    plt.xlabel('face-pair')
    plt.ylabel('HIT')
    plt.title(label)
    # Since we are normalizing response, we use colorbar range of [0,1]
    plt.colorbar()

    fig.savefig(main_dir+'results/'+savename+'.png',dpi=120,facecolor=fig.get_facecolor(),edgecolor='none')
    plt.close(fig)
    
# Loop through HITS, concatenate response vector from each
norm_cat_resp = np.array([normResp(n) for n in range(1,len(Answerface1))])
rank_cat_resp = np.array([rankResp(n) for n in range(1,len(Answerface1))])

# Plot responses
plot_resps(norm_cat_resp,main_dir,'normalized responses','norm_resps')
plot_resps(rank_cat_resp,main_dir,'ranked responses','rank_resps')

# Calculate mean response
mean_resp = norm_cat_resp.mean(axis=0)

def reshape_dsm(mean_resp,FNAMES,FPAIR1):
    """ Reshape response vector into dis-similarity matrix """
    
    # Pre-allocate similarity matrix
    resp_mat = np.ones(shape=(len(FNAMES),len(FNAMES)))*min(mean_resp)

    # Loop through face-pairs
    for i in range(FPAIR1[-1]+1):
       
        # Search out indices of each instance of current pair
        pair_indices = [j for j,x in enumerate(FPAIR1) if x==i]
    
        for ii,x in enumerate(pair_indices):
                resp_mat[ii,i] = mean_resp[x]
                
    return resp_mat
    
resp_mat = reshape_dsm(mean_resp,FNAMES,FPAIR1)


def plot_dsm(resp_mat,FNAMES,FACE_URL,main_dir,label,savename):
    """ Plot dis-similarity matrix with faces as tick-labels """
    
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
        ax1.imshow(face_img)
    
        # Plot on y-axis
        ax2=fig.add_axes([xl-size-0.01, yp+0.02, size, size])
        ax2.axison = False
        ax2.imshow(face_img)
    
    # Create colorbar content
    temp_cb = np.array([range(100) for x in range(3)])
    
    # Plot colorbar manually
    ax1 = fig.add_axes([xl, 0.05, w, h/13.5])
    ax1.imshow(temp_cb,interpolation='nearest',cmap='hot')
    ax1.set_xlabel('mean similarity ('+label+')',color='white')
    ax1.set_xticks((0,100))
    ax1.set_yticks([])
    temp_lims = (round(min(mean_resp)*1000)/1000,round(max(mean_resp)*1000)/1000)
    ax1.set_xticklabels(temp_lims,color='white')
    
    fig.savefig(main_dir+'results/'+savename+'.png',dpi=200,facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    

plot_dsm(resp_mat,FNAMES,FACE_URL,main_dir,'normalized','norm_dsm')

# Calc dis-similarity matrix
dsm = 1-resp_mat
# Classical multidimensional scaling
Y,e = cmdscale(dsm)
# Keep only first and second dimensions (Eigenvalues)
scaled_coords = Y[:,0:2]

def plot_mds(Y,e,FNAMES,main_dir):
    """ Plot faces based on first 2 dimensions of configuration matrix """
    
    fig = plt.figure(figsize=(7,10))
    ax = plt.subplot(211)
    plt.scatter(Y[:,0],Y[:,1])
    for i,txt in enumerate(FNAMES):
        plt.annotate(txt,(Y[i,0],Y[i,1]))

    temp_xlim = ax.get_xlim()
    temp_ylim = ax.get_ylim()

    ax.plot(temp_xlim,[0,0],color='black')
    ax.plot([0,0],temp_ylim,color='black')
    ax.set_xlim(temp_xlim)
    ax.set_ylim(temp_ylim)
    ax.set_xlabel('1st dim. of configuration matrix')
    ax.set_ylabel('2nd dim. of configuration matrix')
    ax.set_title('separation with classical MDS')

    ax = plt.subplot(212)
    ax.plot(e,color='blue')
    temp_xlim = ax.get_xlim()
    ax.plot(temp_xlim,[0,0],color='black')
    ax.set_xlim(temp_xlim)
    ax.set_xlabel('dimension')
    ax.set_ylabel('eigenvalue')
    
    fig.savefig(main_dir+'results/face_mds.png',facecolor='white',edgecolor='none')
    plt.close(fig)

plot_mds(Y,e,FNAMES,main_dir)

pass



