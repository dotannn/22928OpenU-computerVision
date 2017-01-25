import telegram

TOKEN = '313645318:AAFDuySwzhlaD7EUhJ885bB1seta4-D11qo'
bot = telegram.Bot(token=TOKEN)
chat_id = 199392648

import time
import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
import matplotlib.cm as cm
from scipy import interp
from load_data import load_data
from sklearn.metrics.pairwise import euclidean_distances
from skimage import feature
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import scipy.fftpack as fp
from scipy import ndimage
from sklearn.decomposition import SparseCoder
import progressbar
import math
from sklearn.neighbors import KNeighborsClassifier

import cropper; reload(cropper)
import facesclassifier; reload(facesclassifier)
import facepreprocess; reload(facepreprocess)
import facerepresentation; reload(facerepresentation)
import faceoperator; reload(faceoperator)
import frontalizer; reload(frontalizer)

def spatial_hist( img,grid_size_x=8,grid_size_y=8, bins=256):
        h = int(np.ceil(img.shape[0] / grid_size_y))
        w = int(np.ceil(img.shape[1] / grid_size_x))
        hists =[]

        for i in range(0,img.shape[0], h):
            for j in range(0,img.shape[1], w):
                region = img[i:min(i+h,img.shape[0]),j:min(j+w,img.shape[1])]
                hist = np.histogram(region.flatten(), bins=bins, range=(0, bins), normed=True)[0] 

                hists.extend(hist)
        
        return np.asarray(hists).ravel()

def spatial_hist2( img,grid_size_x=8,grid_size_y=8, bins=256, maxVal=0.2):
        h = int(np.ceil(img.shape[0] / grid_size_y))
        w = int(np.ceil(img.shape[1] / grid_size_x))
        hists =[]

        for i in range(0,img.shape[0], h):
            for j in range(0,img.shape[1], w):
                region = img[i:min(i+h,img.shape[0]),j:min(j+w,img.shape[1])]
                hist = np.histogram(region.flatten(), bins=bins, range=(0, bins), normed=False)[0] 
                # normalize :
                hist = hist / np.sqrt( (hist**2).sum() )
                hist[hist>0.2] = 0.2
                hist = hist / np.sqrt( (hist**2).sum() )
                
                hists.extend(hist)
        
        return np.asarray(hists).ravel()

def homomorphic_filtering(img,high=1.1, low=0.5, D0=15,c=1.2):
    
    # convert image to 0-1 float
    imgf = np.float32(img)/255

    rows = img.shape[0]
    cols = img.shape[1]

    #take log
    logimg= np.log1p(imgf)
    
    #create lowpass & highpass:
    M = rows
    N = cols
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    H = (high-low)* (1-np.exp(-c* gaussianNumerator / (2*D0*D0))) + low
    
    
    Hshift= fp.ifftshift(H.copy())
    
    # Filter the image and crop
    fftImg = fp.fft2(logimg.copy())
#    Ioutlow = np.real(fp.ifft2(If.copy() * HlowShift, (M,N)))
#    Iouthigh = np.real(fp.ifft2(If.copy() * HhighShift, (M,N)))

    # Set scaling factors and add
    Iout = np.real(fp.ifft2(fftImg*Hshift))

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")
    return Ihmf2
        
        
def fixface (img):
    ret = img
    width = img.shape[1]
    e_left=  img[:,:width/2].sum()
    e_right=  img[:,width/2:].sum()
    
    ratio = e_left/ e_right
    if  ratio < 0.75 :
        ret[:,:width/2] = np.fliplr(img)[:,:width/2]
    elif ratio > 1.25 :
        ret[:,width/2:] = np.fliplr(img)[:,width/2:]
        
    return ret
    
def TanTriggs( img, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1.0, sigma1 = 2.0):
    # convert to float32
    X = np.float32(img)
        
    # gamma correct
    I = np.power(X,gamma)
    # calc DoG:
    dog = np.asarray(ndimage.gaussian_filter(I,sigma1) - ndimage.gaussian_filter(I, sigma0))
        
    dog = dog / np.power(np.mean(np.power(np.abs(dog), alpha)), 1.0/ alpha)
    dog = dog / np.power(np.mean(np.power(np.minimum(np.abs(dog), tau), alpha)), 1.0/alpha)
    dog =  tau*np.tanh(dog/tau)
        
    # normalize
    return cv2.normalize(dog,dog,0,255,cv2.NORM_MINMAX, cv2.CV_8UC1)
        

def chi2_d(a, b):
    eps = 1e-10    
       # compute the chi-squared distance
    return 0.5 * np.sum([((a- b) ** 2) / (a+ b+ eps)])
    
def chi2_distance(histA, refs):
    eps = 1e-10
        
    d = np.zeros((len(refs),1))
        
    for idx in range(len(refs)):
        histB = refs[idx]
        # compute the chi-squared distance
        d[idx] = 0.5 * np.sum([((histA - histB) ** 2) / (histA + histB + eps)])
    return d

def makePatchSampleCoordMatrix(YY,XX,rows,cols,patchRadius):
    # sample patch for each circle center :
    ind = XX*rows + YY
    
    [x,y] = np.meshgrid(range(-patchRadius,patchRadius+1), range(-patchRadius,patchRadius+1))
    x = x.ravel('F')
    y = y.ravel('F')
    
    offsets = x + y*rows;
    
    r_ind = np.matlib.repmat(ind,len(offsets),1).transpose()
    r_offsets = np.matlib.repmat(offsets,len(ind),1)
    
    # sample each circle center
    indSample = r_ind + r_offsets
    return np.int32(indSample)
    
def tplbp(img, w=3, radius=2, S=8, alpha=5, tau=0.01):
    imgf = np.float64(img)
    
    patchRadius = int(math.floor(w/2))
    rows = img.shape[0]
    cols = img.shape[1]
    border = radius + 2*patchRadius
    [XXbase,YYbase]=np.meshgrid(range( cols),range(rows));
    
    XX = XXbase[border:-border,border:-border].ravel('F')
    YY = YYbase[border:-border,border:-border].ravel('F')
    ii= XX*rows + YY
    
    indSample = makePatchSampleCoordMatrix(YY,XX,rows,cols,patchRadius)
    centerPatches = imgf.ravel('F')[indSample]


    # deltas around circle center
    angles =np.linspace(-np.pi,np.pi,9)
    angles = angles[:-1]
    ysamples = radius* np.sin(angles)
    xsamples = radius* np.cos(angles)
    
    # XX ? maybe a bug
    Ydeltas =np.matlib.repmat(ysamples,len(XX),1)
    Xdeltas =np.matlib.repmat(xsamples,len(XX),1)
    
    XXrep = np.matlib.repmat(XX,S,1).transpose()
    YYrep = np.matlib.repmat(YY,S,1).transpose()
    
    XXcircles = XXrep + Xdeltas
    YYcircles = YYrep + Ydeltas
    
    XXcircles = np.round(XXcircles);
    YYcircles = np.round(YYcircles);

    D = np.zeros(XXcircles.shape)
    
    for ni in range(S):
        indSample = makePatchSampleCoordMatrix(YYcircles[:,ni],XXcircles[:,ni], rows, cols, patchRadius)
        neighborPatches = imgf.ravel('F')[indSample]

        D[:,ni] = np.sum((neighborPatches-centerPatches)**2,1)
    
    D2 = np.roll(D,-alpha,1)
    Dsub = D - D2
    Code = Dsub > tau
    codeI = np.zeros(img.shape[0]* img.shape[1],'uint8')
    for bit in range(S):
        bitsInds = np.where(Code[:,bit])[0]
        codeI[ii[bitsInds]] += 2**bit
    
    return codeI.reshape((rows,cols),order='F')
        #http://www.openu.ac.il/home/hassner/projects/Patchlbp/WolfHassnerTaigman_ECCVW08.pdf
#    res = np.zeros(img.shape, 'uint8')
#    
#    
#    
#    for i in range(radius,img.shape[0]-radius):
#        for j in range(radius,img.shape[1]-radius):        
#            bytestr =  (np.sign((img[i,j] - img[i  ,j-radius])) + 1)  *0.5 +\
#                       (np.sign((img[i,j] - img[i-radius,j-radius])) + 1)       +\
#                       (np.sign((img[i,j] - img[i-radius,j  ])) + 1)  * 2+\
#                       (np.sign((img[i,j] - img[i-radius,j+radius])) + 1)  * 4+\
#                       (np.sign((img[i,j] - img[i  ,j+radius])) + 1)  * 8+\
#                       (np.sign((img[i,j] - img[i+radius,j+radius])) + 1)  * 16+\
#                       (np.sign((img[i,j] - img[i+radius,j  ])) + 1)  * 32+\
#                       (np.sign((img[i,j] - img[i+radius,j-radius])) + 1)  * 64
#            print (bytestr)
#            res[i,j] = bytestr
                
#    return res
        
basedir = '/media/dotan/Data/datasets/proj/'
n_train = 65

train_data, test_data, labelnames = load_data(basedir + 'ExtendedYaleB/',n_train)
#
train_imgs, ytrain = zip(*train_data)
test_imgs, ytest = zip(*test_data)

#frn = frontalizer.FaceFrontalizer('rotation')
                                                  
#crop = cropper.Cropper((100,100),'resize','affine')
#
#train_cropped_imgs = crop.crop(train_imgs)
#test_cropped_imgs = crop.crop(test_imgs)
#print 'calc train lbp'
#train_lbps =[]
#for img in train_cropped_imgs:
#    train_lbps.append(feature.local_binary_pattern(homomorphic_filtering(img),8,2, method='ror').astype('uint8'))
#
#    
#print 'calc test lbp'
#test_lbps = []
#for img in test_cropped_imgs:
#    test_lbps.append(feature.local_binary_pattern(homomorphic_filtering(img),8,2, method='ror').astype('uint8'))
#
#
#for grid_x in range(3,5):
#    for grid_y in range(3,5):
#        train_hists =[]
#        test_hists = []
#        print 'calc hists for '+str(grid_y) + 'x' + str(grid_x)+' :'
#        for lbp in train_lbps :
#            train_hists.append(spatial_hist(lbp, grid_size_x=(grid_x), grid_size_y=(grid_y) ))
#            
#        for lbp in test_lbps :
#            test_hists.append(spatial_hist(lbp, grid_size_x=(grid_x) , grid_size_y=(grid_y) ))
#            
#        ypredict = np.zeros((len(test_hists),1),'int')
#        for idx in range(len(test_hists)):
#            dist = chi2_distance(test_hists[idx], train_hists)
#            yIdx = np.argmin(dist)
#            ypredict[idx] = ytrain[yIdx]
#   
#
#        text= "CROPPED 100X100, affine alignment,fixface, lbp radius = 1, grid " + str(grid_y) + "x" + str(grid_x) +": " + str(metrics.accuracy_score(ytest,ypredict))
#        bot.sendMessage(chat_id=chat_id, text=text)
#        
##
##train_cropped_imgs = crop.crop(train_imgs)
###
##test_cropped_imgs = crop.crop(test_imgs)
#
#
## check what happen when resizing by half
#
#facerec = cv2.face.createLBPHFaceRecognizer()
#train_resized_imgs =[]
#for img in train_imgs:
#    train_resized_imgs.append(cv2.resize(img,dsize=None, fx=0.25, fy=0.25))
#
#test_resized_imgs = []
#for img in test_imgs:
#    test_resized_imgs.append(cv2.resize(img,dsize=None,fx=0.25, fy=0.25))
#    
#    
#facerec.train(train_resized_imgs,np.int32(ytrain))
#
#ypredict = []
#for img in test_imgs :
#    ypredict.append(facerec.predict(img))
#        
#text= "check what happen when resizing by quater : " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)


# do the same with local_binary_pattern with different structure
#
#train_hists =[]
#for img in train_imgs:
#    lbp = feature.local_binary_pattern(img,8,2 , method='ror')
#    train_hists.append(spatial_hist(lbp, grid_size_x=2, grid_size_y=2,bins=256))
#
#test_hists = []
#for img in test_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    test_hists.append(spatial_hist(lbp,grid_size_x=2, grid_size_y=2,bins=256))
#    
#
#ypredict = np.zeros((len(test_hists),1),'int')
#
#for idx in range(len(test_hists)):
#    dist = chi2_distance(test_hists[idx], train_hists)
#    yIdx = np.argmin(dist)
#    ypredict[idx] = ytrain[yIdx]        
#
#
#text= "do lbp r=2 p=8, rotation invariant hist, grid 2x2: " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#
#
#train_hists =[]
#for img in train_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    train_hists.append(spatial_hist(lbp))
#
#test_hists = []
#for img in test_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    test_hists.append(spatial_hist(lbp))
#    
#    
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#svmModel = GridSearchCV(svm.SVC(kernel='linear', class_weight='balanced'), param_grid)
#
#train_hists = np.vstack(train_hists)
#svmModel.fit(train_hists,ytrain)
#
#test_hists = np.vstack(test_hists)
#ypredict = svmModel.predict(test_hists)
#
#text= "lbp hists then linear svm: " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#
#
## do default but with 'ror'
#train_hists =[]
#for img in train_imgs:
#    lbp = feature.local_binary_pattern(img,8,1, method='ror')
#    train_hists.append(spatial_hist(lbp))
#
#test_hists = []
#for img in test_imgs:
#    lbp = feature.local_binary_pattern(img,8,1,method='ror')
#    test_hists.append(spatial_hist(lbp))
#    
#
#ypredict = np.zeros((len(test_hists),1),'int')
#
#for idx in range(len(test_hists)):
#    dist = chi2_distance(test_hists[idx], train_hists)
#    yIdx = np.argmin(dist)
#    ypredict[idx] = ytrain[yIdx]        
#
#
#text= "do the same (nothing) with ror: " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#
#
#
## do default but with 'uniform'
#train_hists =[]
#for img in train_imgs:
#    lbp = feature.local_binary_pattern(img,8,1, method='nri_uniform')
#    train_hists.append(spatial_hist(lbp,bins=59))
#
#test_hists = []
#for img in test_imgs:
#    lbp = feature.local_binary_pattern(img,8,1,method='nri_uniform')
#    test_hists.append(spatial_hist(lbp,bins=59))
#    
#
#ypredict = np.zeros((len(test_hists),1),'int')
#
#for idx in range(len(test_hists)):
#    dist = chi2_distance(test_hists[idx], train_hists)
#    yIdx = np.argmin(dist)
#    ypredict[idx] = ytrain[yIdx]        
#
#
#text= "do the same (nothing) with nri_uniform: " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#
#
#
#train_hists =[]
#for img in train_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    train_hists.append(spatial_hist(lbp, grid_size_x=10, grid_size_y=10))
#
#test_hists = []
#for img in test_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    test_hists.append(spatial_hist(lbp,grid_size_x=10, grid_size_y=10))
#    
#
#ypredict = np.zeros((len(test_hists),1),'int')
#
#for idx in range(len(test_hists)):
#    dist = chi2_distance(test_hists[idx], train_hists)
#    yIdx = np.argmin(dist)
#    ypredict[idx] = ytrain[yIdx]        
#
#
#
##param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
##              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
##svm = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
#
#text= "do the same (nothing) lbp radius = 2 + ror, grid 10x10: " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#
#
#train_hists =[]
#for img in train_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    train_hists.append(spatial_hist(lbp, grid_size_x=8, grid_size_y=8))
#
## learn some similar labels :
#svm4_24 = svm.SVC(kernel='linear', C=1000)
#train4_24 = [train_hists[i] for i in range(len(ytrain)) if ytrain[i]==4 ] + [train_hists[i] for i in range(len(ytrain)) if ytrain[i]==24 ]             
#train4_24 = np.vstack(train4_24)
#
#ytrain4_24 = [0]*65 + [1]*65
#svm4_24.fit(train4_24, ytrain4_24)
#
#test_hists = []
#for img in test_imgs:
#    lbp = feature.local_binary_pattern(img,8,2, method='ror')
#    test_hists.append(spatial_hist(lbp,grid_size_x=8, grid_size_y=8))
#    
#
#ypredict = np.zeros((len(test_hists),1),'int')
#
#for idx in range(len(test_hists)):
#    dist = chi2_distance(test_hists[idx], train_hists)
#    yIdx = np.argmin(dist)
#    ypredict[idx] = ytrain[yIdx]
#    if ypredict[idx] ==4 :
#        if svm4_24.predict([test_hists[idx]])[0] != 0 :
#            ypredict[idx] = 24
#
#
#text= "do the same (nothing) lbp radius = 2 + ror, grid 8x8, svm on 4-24: " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#
#
#
#train_hists =[]
#for img in train_imgs:
#    lbp = tplbp(img,radius=2)
#    lbp = lbp[2:-2,2:-2]
#    train_hists.append(spatial_hist(lbp, grid_size_x=8, grid_size_y=8))
#
#
#test_hists = []
#for img in test_imgs:
#    lbp = tplbp(img,radius=2)
#    lbp = lbp[2:-2,2:-2]
#    test_hists.append(spatial_hist(lbp,grid_size_x=8, grid_size_y=8))
#    
#
#ypredict = np.zeros((len(test_hists),1),'int')
#
#for idx in range(len(test_hists)):
#    dist = chi2_distance(test_hists[idx], train_hists)
#    yIdx = np.argmin(dist)
#    ypredict[idx] = ytrain[yIdx]
#   
#
#text= "tplbp radius = 2, grid 8x8 " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)
#



# check grid size :
#print 'Check grid size...'
#
#print 'calc train lbp'
#train_lbps =[]
#for img in train_imgs:
#    train_lbps.append(feature.local_binary_pattern(img,8,2, method='ror').astype('uint8'))
#
#    
#print 'calc test lbp'
#test_lbps = []
#for img in test_imgs:
#    test_lbps.append(feature.local_binary_pattern(img,8,2, method='ror').astype('uint8'))
#
#print 'finish with lbp'    
#for grid_x in range(8,10):
#    for grid_y in reversed(range(10)):
#        train_hists =[]
#        test_hists = []
#        print 'calc hists for '+str(grid_y) + 'x' + str(grid_x)+' :'
#        for lbp in train_lbps :
#            train_hists.append(spatial_hist(lbp, grid_size_x=(grid_x+1), grid_size_y=(grid_y+1) ))
#            
#        for lbp in test_lbps :
#            test_hists.append(spatial_hist(lbp, grid_size_x=(grid_x+1) , grid_size_y=(grid_y+1) ))
#            
#        ypredict = np.zeros((len(test_hists),1),'int')
#        for idx in range(len(test_hists)):
#            dist = chi2_distance(test_hists[idx], train_hists)
#            yIdx = np.argmin(dist)
#            ypredict[idx] = ytrain[yIdx]
#   
#
#        text= "lbp radius = 2, grid " + str(grid_y) + "x" + str(grid_x) +": " + str(metrics.accuracy_score(ytest,ypredict))
#        bot.sendMessage(chat_id=chat_id, text=text)
#


#best alg :
#    
#
#grid_x = 8
#grid_y= 7
#
#train_hists =[]
#test_hists = []
#
#bar = progressbar.ProgressBar()
#print ('calc train features')
#for imgIdx in bar(range(len(train_imgs))):
#    train_hists.append(spatial_hist(feature.local_binary_pattern(train_imgs[imgIdx],8,2, method='ror').astype('uint8'), grid_size_x=grid_x, grid_size_y=grid_y))
#
#bar = progressbar.ProgressBar()
#print ('calc test features')
#for imgIdx in bar(range(len(test_imgs))):
#    test_hists.append(spatial_hist(feature.local_binary_pattern(test_imgs[imgIdx],8,2, method='ror').astype('uint8'), grid_size_x=grid_x, grid_size_y=grid_y))
#
#    
#
#knn=KNeighborsClassifier(n_neighbors=1,
#                 algorithm='auto',
#                 metric=lambda a,b: chi2_d(a,b)
#                 )
#
#knn.fit(np.vstack(train_hists), ytrain)
#ypredict = knn.predict(np.vstack(test_hists))
#
#text= "BEST?  lbp radius = 2, grid " + str(grid_y) + "x" + str(grid_x) +": " + str(metrics.accuracy_score(ytest,ypredict))
#bot.sendMessage(chat_id=chat_id, text=text)


# test tplbp with crop
#    
#
grid_x = 11
grid_y= 13

train_hists =[]
test_hists = []

crop = cropper.Cropper((180,180),'resize','affine')

train_cropped_imgs = crop.crop(train_imgs)
test_cropped_imgs = crop.crop(test_imgs)

bar = progressbar.ProgressBar()
print ('calc train features')
for imgIdx in bar(range(len(train_imgs))):
    train_hists.append(spatial_hist2(tplbp(train_imgs[imgIdx]), grid_size_x=grid_x, grid_size_y=grid_y))

bar = progressbar.ProgressBar()
print ('calc test features')
for imgIdx in bar(range(len(test_imgs))):
    test_hists.append(spatial_hist2(tplbp(test_imgs[imgIdx]), grid_size_x=grid_x, grid_size_y=grid_y))

    
print ('calc knn')
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(np.vstack(train_hists), ytrain)
print ('predicting....')
ypredict = knn.predict(np.vstack(test_hists))

text= "test TPLBP cropped 180x180 l2-dist, grid " + str(grid_y) + "x" + str(grid_x) +": " + str(metrics.accuracy_score(ytest,ypredict))
bot.sendMessage(chat_id=chat_id, text=text)
