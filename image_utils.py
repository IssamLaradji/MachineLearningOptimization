import pickle
import os
import sys
import numpy as np
import pylab as plt 
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
from skimage.transform import rescale
from PIL import Image, ImageStat



def mask_and_slide(img, mask, mask_bs=80, slide_bs=200, stride=200):
    img_tmp = img.copy()
    mask_tmp = mask.copy()

    indices = np.where(np.squeeze(mask)==1)

    crList = indices[0]
    ccList = indices[1]

    for cr, cc in zip(crList, ccList):
        sr = max(cr - mask_bs/2, 0)
        sc = max(cc - mask_bs/2, 0) 

        img_tmp[sr:sr+mask_bs,sc:sc+mask_bs] = 0.
        mask_tmp[sr:sr+mask_bs,sc:sc+mask_bs] = 0.

    Xp = get_sliding_single(img_tmp, bs=slide_bs, stride=stride)
    Yp = get_sliding_single(mask_tmp, bs=slide_bs, stride=stride)

    return (Xp, Yp)


def get_sliding_single(img, bs=200, stride=200):
    if img.ndim == 2:
        img = img[:,:,np.newaxis]

    n_rows, n_cols, n_channels = img.shape
    Xp = np.zeros(((n_rows/stride)*(n_cols/stride), bs, bs, n_channels))
    j = 0
    
    for sr in range(0, n_rows, stride):
        for sc in range(0, n_cols, stride):
            try:
                Xp[j] = img[sr:sr+bs, sc:sc+bs]
                j += 1
            except:
                pass

    return Xp[:j]

def get_sliding(X, bs=200, stride=200):
    n, c, n_rows, n_cols = X.shape
    Xp = np.zeros((n*(n_rows/stride)*(n_cols/stride), c, bs, bs))
    j = 0
    for i in range(n):
        for sr in range(0, n_rows, stride):
            for sc in range(0, n_cols, stride):
                try:
                    Xp[j] = X[i, :, sr:sr+bs, sc:sc+bs]
                    j += 1
                except:
                    pass

    return Xp[:j]

def showm(img):
    img = np.squeeze(img)
        
    if img.ndim == 3:
        plt.imshow(np.transpose(img, [1, 2, 0]).mean(axis=2))
    else:
        plt.imshow(img, cmap=plt.get_cmap('gray'))

    plt.show()

def brightness(img):
    rgb = img.mean(axis=0).mean(axis=0)
    bright = np.sqrt(np.dot([0.241, 0.691, 0.068], rgb**2))

    return bright

def adjust_brightness(img, offset):
    img += offset


    return np.clip(img, 0., 1.)

def read_image(filename, rule="rgb", scale=None, shape=None):
    if rule == "rgb":
        img = imread(filename)

    if scale is not None:
        return rescale(img, scale)
    if shape is not None:
        return resize(img, shape)

    return img_as_float(img)

def adjust_contrast(img, offset):
    img *= offset

    return np.clip(img, 0., 1.)

def single(img, scale=1):
    img = np.squeeze(img)
    if img.ndim == 4:
        img = np.squeeze(img)

    if img.ndim == 3 and img.shape[0] == 3:     
        img = first2last(img)


    if scale != 1:
        img = rescale(img, scale)

    if img.ndim == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=plt.get_cmap('gray'))

def show(*imgList):
    N = len(imgList)
    if N == 1:
        single(imgList[0])
    else:

        for i, img in enumerate(imgList):
            if N > 3:
                plt.subplot(1 + N/3, 3, i+1)
            else:
                plt.subplot(N, 1, i+1)

            single(img)

    plt.tight_layout()
    plt.show() 

def save(*imgList):
    N = len(imgList)
    if N == 1:
        single(imgList[0])
    else:

        for i, img in enumerate(imgList):
            if N > 3:
                plt.subplot(1 + N/3, 3, i+1)
            else:
                plt.subplot(N, 1, i+1)

            single(img)

    plt.tight_layout()
    save = "plt_saved.png"
    plt.savefig(save)
    print "%s saved..." % save
    plt.close()


def show_scaled(scale=0.2, *imgList):
    N = len(imgList)
    if N == 1:
        single(imgList[0], scale)
    else:

        for i, img in enumerate(imgList):
            if N > 3:
                plt.subplot(1 + N/3, 3, i+1)
            else:
                plt.subplot(N, 1, i+1)

            single(img, scale)

    plt.tight_layout()
    plt.show() 

def orient(img):
    img = np.squeeze(img)
    if img.ndim == 4:
        img = np.squeeze(img)

    if img.ndim == 3 and img.shape[0] == 3:
       return np.transpose(img, [1, 2, 0])
    
    return img


def show_heat(model, img, colorbar=False, show=True, save=False):
    show_mask(img, model.predict(img), colorbar=colorbar, 
                   show=show, save=save)
    
def show_mask(img, mask, colorbar=False, show=True, save=False):
    img = orient(img)
    mask = orient(mask)
    
    if mask.shape[:2] != img.shape[:2]:
        mask = resize(mask, img.shape[:2])

    plt.imshow(img); plt.imshow(mask, alpha=0.5)
    if colorbar:
        plt.colorbar()

    if show:
        plt.show()
        print

    if save:
        plt.savefig(save)
        print "%s saved..." % save



def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def first2last(X):
    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1,2,0))
    if X.ndim == 4:
        return np.transpose(X, (0,2,3,1))

def last2first(X):
    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2,0,1))
    if X.ndim == 4:
        return np.transpose(X, (0,3,1,2))

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


