# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:24:05 2017

@author: lab
"""

import xlrd, os, urllib, cStringIO
import numpy as np
import pylab as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.stats import rankdata
from cmdscale import cmdscale

# Define path to git repo
MAIN_DIR = os.environ['HOME']+"/Cloud2/movies/human/turk/face_sim/"

FNAMES = ['ArnoldBarney','BarneyDaniel','DanielHillary','DanielShinzo','HillaryShinzo','IanPiers','IanTom','PiersTom']   
FPAIR1 = [0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6]
FPAIR2 = [1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7]

FACE_URL = ['http://i.imgur.com/HFMr5Jp.png','http://i.imgur.com/VoB4Hr1.png','http://i.imgur.com/8afrTze.png','http://i.imgur.com/vvvahSz.png','http://i.imgur.com/xZLTaKf.png','http://i.imgur.com/M1g57XX.png','http://i.imgur.com/XA8A3FR.png','http://i.imgur.com/Jbh4AeB.png']

# import data
workbook = xlrd.open_workbook(MAIN_DIR+'results/test_batch.xlsx')
sheet = workbook.sheet_by_index(0)

# Allocate imported worksheet to column variable names
FACE_PAIR = sheet.col_values(2)
RESPONSES = sheet.col_values(3)

def main():
    # Loop through HITS, concatenate response vector from each
    norm_cat_resp = np.array([normResp(n) for n in range(1,len(FACE_PAIR))])
    rank_cat_resp = np.array([rankResp(n) for n in range(1,len(FACE_PAIR))])
    
    # Plot responses
    plot_resps(norm_cat_resp,MAIN_DIR,'normalized responses','norm_resps')
    plot_resps(rank_cat_resp,MAIN_DIR,'ranked responses','rank_resps')
    
    # Calculate mean response
    norm_mean_resp = norm_cat_resp.mean(axis=0)
    rank_mean_resp = rank_cat_resp.mean(axis=0)
    
    # Reshape mean responses into similarity matrix
    norm_resp_mat = reshape_dsm(norm_mean_resp,FNAMES,FPAIR1)
    rank_resp_mat = reshape_dsm(rank_mean_resp,FNAMES,FPAIR1)
    
    # Plot similarity matrix
    plot_dsm(norm_resp_mat,FNAMES,FACE_URL,MAIN_DIR,'normalized','norm_dsm')
    plot_dsm(rank_resp_mat,FNAMES,FACE_URL,MAIN_DIR,'ranked','rank_dsm')
    
    # Calc dis-similarity matrix
    norm_dsm = 1-norm_resp_mat
    rank_dsm = 1-rank_resp_mat
    
    # Classical multidimensional scaling
    norm_config_vals,norm_eigvals = cmdscale(norm_dsm)
    rank_config_vals,rank_eigvals = cmdscale(rank_dsm)
    
    # Plot MDS results
    plot_mds(norm_config_vals,norm_eigvals,FNAMES,MAIN_DIR,'normalized','norm_mds')
    plot_mds(rank_config_vals,rank_eigvals,FNAMES,MAIN_DIR,'ranked','rank_mds')
    
    # Create list of image filenames 
    img_list = [MAIN_DIR+'results/'+x+'.png' for x in ['norm_resps', 'norm_dsm', 'norm_mds']]
    
    # Load images into list and collect sizes
    images = map(Image.open, img_list)
    widths, heights = zip(*(i.size for i in images))
    
    # Create new (scaled) sizes
    new_height = 800.0
    new_widths = widths*np.array([new_height/x for x in heights])
    
    # Create new image for others to be pasted into
    new_im = Image.new('RGB', (int(sum(new_widths)), int(new_height)))
        
    x_offset = 0
        
    for j, im in enumerate(images):
        # Scale and paste  
        scaled_im = im.resize((new_widths.astype(int)[j],int(new_height)))
        new_im.paste(scaled_im, (x_offset,0))
        x_offset += scaled_im.size[0]
        
    new_im.save(MAIN_DIR+'results/summary.png')

def normResp(hit_num):
    """De-randomize, normalize, and concatenate responses for each HIT"""
    
    # Convert strings to ints
    responses = np.array(map(int, RESPONSES[hit_num].split(',')))
    order = np.array(map(int, FACE_PAIR[hit_num].split(',')))

    # De-randomize
    derand_resp = responses[order]

    # Normalize b/t 0 and 1
    norm_resp = derand_resp - derand_resp.min()    
    norm_resp = norm_resp/float(norm_resp.max())
    
    return norm_resp
    
def rankResp(hit_num):
    """De-randomize, rank-order, and concatenate responses for each HIT"""
    
    # Convert strings to ints
    responses = np.array(map(int, RESPONSES[hit_num].split(',')))
    order = np.array(map(int, FACE_PAIR[hit_num].split(',')))

    # De-randomize
    derand_resp = responses[order]

    # Rank-order responses
    rank_resp = rankdata(derand_resp)
#    rank_resp = norm_resp/float(norm_resp.max())
    
    return rank_resp

def plot_resps(cat_resp,MAIN_DIR,label,savename):
    """ Plot responses """
    
    fig, ax = plt.subplots(1,figsize=(5,11),facecolor='white')
    
    plt.subplot(111)
    plt.imshow(cat_resp,interpolation='none',cmap='hot')
    plt.xlabel('face-pair')
    plt.ylabel('HIT')
    plt.title(label)
    # Since we are normalizing response, we use colorbar range of [0,1]
    plt.colorbar()

    fig.savefig(MAIN_DIR+'results/'+savename+'.png',dpi=120,facecolor=fig.get_facecolor(),edgecolor='none')
    plt.close(fig)
    
def reshape_dsm(mean_resp,FNAMES,FPAIR1):
    """ Reshape response vector into dis-similarity matrix """
    
    # Pre-allocate similarity matrix
    resp_mat = np.ones(shape=(len(FNAMES),len(FNAMES)))*min(mean_resp)

    # Loop through face-pairs
    for i in range(FPAIR1[-1]+1):

        # Search out indices of each instance of current pair
        pair_indices = [j for j,x in enumerate(FPAIR1) if x==i]
    
        for ii,x in enumerate(pair_indices):
                resp_mat[FPAIR1[-1]+1-ii,i] = mean_resp[x]
                resp_mat[i,FPAIR1[-1]+1-ii] = mean_resp[x]
                
    return resp_mat
    
def plot_dsm(resp_mat,FNAMES,FACE_URL,MAIN_DIR,label,savename):
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
        ax2=fig.add_axes([xl-size-0.01, yh-yp, size, size])
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
    temp_lims = (round(resp_mat.min()*1000)/1000,round(resp_mat.max()*1000)/1000)
    ax1.set_xticklabels(temp_lims,color='white')
    
    fig.savefig(MAIN_DIR+'results/'+savename+'.png',dpi=200,facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    
def plot_mds(config_vals,eigen_vals,FNAMES,MAIN_DIR,label,savename):
    """ Plot faces based on first 2 dimensions of configuration matrix """
    
    fig = plt.figure(figsize=(7,10))
    ax = plt.subplot(211)
    
    crop_size = 20

    # Loop through face images
    for i,x in enumerate(FACE_URL):

        plt.scatter(config_vals[i,0],config_vals[i,1])
    
        # Load image using URL
        file = cStringIO.StringIO(urllib.urlopen(x).read())
        face_img = Image.open(file)
    
        # Crop image edges
        face_img = face_img.crop((crop_size,crop_size,face_img.size[0]-crop_size,face_img.size[1]-crop_size))
    
        # Plot images as points
        imscatter(config_vals[i,0],config_vals[i,1],face_img,zoom=0.5,ax=ax)

        # Add face-names to each point
        plt.annotate(FNAMES[i],(config_vals[i,0]+config_vals.max()/8,config_vals[i,1]))

    temp_xlim = ax.get_xlim()
    temp_ylim = ax.get_ylim()

    ax.plot(temp_xlim,[0,0],color='black')
    ax.plot([0,0],temp_ylim,color='black')
    ax.set_xlim(temp_xlim)
    ax.set_ylim(temp_ylim)
    ax.set_xlabel('1st dim. of configuration matrix')
    ax.set_ylabel('2nd dim. of configuration matrix')
    ax.set_title('separation of '+label+' responses with classical MDS')

    # Plot eigenvalues
    ax = plt.subplot(212)
    ax.plot(eigen_vals,color='blue')
    temp_xlim = ax.get_xlim()
    ax.plot(temp_xlim,[0,0],color='black')
    ax.set_xlim(temp_xlim)
    ax.set_xlabel('dimension')
    ax.set_ylabel('eigenvalue')
    
    # Save plot to figure and close
    fig.savefig(MAIN_DIR+'results/'+savename+'.png',dpi=200,facecolor='white',edgecolor='none')
    plt.close(fig)
    
def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
        
    img = image.convert("RGBA")
    datas = img.getdata()
    
    # Convert black pixels to alpha (transparent)
    newData = []
    for item in datas:
        if item[0] < 5 and item[1] < 5 and item[2] < 5:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)
    
    img.putdata(newData)

    im = OffsetImage(img, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

main()


