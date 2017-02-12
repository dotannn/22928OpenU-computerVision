#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:31:15 2017

Face Recognition class, contains all the stages and methods implemented
for the project.

@author: dotan
"""

# ipmort dependencies
import dlib
import math
import cv2
import numpy as np
from sklearn import svm
from skimage import feature
import scipy.fftpack as fp
from scipy import ndimage
import progressbar
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
import dill # this is used to pickle lambda functions
import pickle

# Template for affine alignment describing the canonical form of face representation
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

class FaceRecotnition:
    def __init__(self, in_method='none', pn_method='none',\
                 crop=(-1,-1), features='lbph_ror', S=8, radius=2, grid_x=8, grid_y=7,\
                 hist_norm='default', classifier='nn_chi2', C=200,finetune=1):
        """
        Class c'str
    
        getting configuration and initialize helper classes
    
        Parameters
        ----------
        in_method : string
            illumination normalization method.
            possible values: 'tantriggs', 'homomorphic', 'equalizehist', 'none'
            default : 'none'
            
        pn_method : string
            pose normalization(alignment) method
            possible values: 'affine', 'rotate', 'none'
            default : 'none'
            
        crop : (int,int)
            face crop resolution, when set to (-1,-1) no cropping is done
            default: (-1,-1)
            
        features : string
            feature extractor methods
            possible values: 'lbpror_h', 'lbp_h', 'tplbp_h', 'tplbp_lbpror_h'. 'ltp_h'
            default : 'lbpror_h'
            
        S : int :
            neighbors for lbp and its variant (tplbp)
            default: 8
            
        radius : int
            lbp/tplbp radius
            default:2
            
        grid_x : int
            with grid_y will deside how many different patches will be used for
            the spatial hist
            default : 8
            
        grid_y : int
            see grid_x
            default : 7
            
        hist_norm : string
            spatial-histogram normalization method.
            possible values: 'default' - standard normalization,
                             'hellinger' - normalization preparing to l2 compare
                             'hassner' - normalization found in "Descriptor Based Methods in the Wild"
                             
        classifier :string
            classifier type
            possible values: 'nn_chi2' one nearest neighbor with chi2-dist metric
                             'nn_l2' -  one nearest neighbor with l2 metric
                             'linearsvm' - Multiclass linear svm
                             'rbfsvm' - Multiclass rbf svm.
  
        C: float
            regularization term for SVM
        finetune: bool
            flag indicates whether to do finetune on specific pairs
        """  
        self.in_method= in_method
        self.pn_method= pn_method
        self.crop= crop
        self.features= features
        self.radius= radius
        self.S = S
        self.grid_x= grid_x
        self.grid_y= grid_y
        self.hist_norm =hist_norm
        self.classifier_type = classifier
        self.C = C
        self.eps = 1e-7
        self.finetune = finetune
        print('v0.19')
        
        self.cascades = []
        self.cascades.append(dlib.get_frontal_face_detector())
        self.cascades.append(cv2.CascadeClassifier('./lbpcascade_frontalface.xml'))
        self.cascades.append(cv2.CascadeClassifier('./lbpcascade_profileface.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_default.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_frontalface_alt_tree.xml'))
        self.cascades.append(cv2.CascadeClassifier('./haarcascade_profileface.xml'))
        
        self.landmarkDetector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.clahe = cv2.createCLAHE( clipLimit=10.0,  tileGridSize=(12,12))
        # init classifier : 
        if (self.classifier_type == 'nn_chi2' ):
            self.classifier = KNeighborsClassifier(n_neighbors=1,
                     algorithm='auto',
                     metric=lambda a,b: self.chi2_d(a,b))
        elif (self.classifier_type == 'linearsvm'):
            self.classifier = OneVsRestClassifier(svm.SVC(probability=True, kernel='linear', C=self.C))
        elif (self.classifier_type == 'rbfsvm'):
            self.classifier = OneVsOneClassifier(svm.SVC(probability=True, kernel='rbf', C=self.C))
        else:
            self.classifier = KNeighborsClassifier(n_neighbors=1)
            
        if self.finetune:
            self.classifier7_5 = svm.SVC(probability=True, kernel='linear', C=self.C)
            self.classifier25_22 = svm.SVC(probability=True, kernel='linear', C=self.C)
            
    def get_config(self):
        return 'FACE-REC  in_method: ' + self.in_method + ', pn_method:'+\
            self.pn_method +', crop:'+str(self.crop) + ', desc:'+ self.features+\
            ', S:' + str(self.S) + ', Radius:' + str(self.radius) + ', grid:('+str(self.grid_y) + ','+str(self.grid_x)+\
            '), hist-norm:' + self.hist_norm + ', classifier: ' + self.classifier_type + ', finetune:' + str(self.finetune)
    
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['clahe']
        del d['cascades']
        del d['landmarkDetector']
        return d
        
    def __setstate__(self, d):
        d['clahe'] = cv2.createCLAHE( clipLimit=10.0,  tileGridSize=(12,12))
        d['cascades'] = []
        d['cascades'].append(dlib.get_frontal_face_detector())
        d['cascades'].append(cv2.CascadeClassifier('./lbpcascade_frontalface.xml'))
        d['cascades'].append(cv2.CascadeClassifier('./lbpcascade_profileface.xml'))
        d['cascades'].append(cv2.CascadeClassifier('./haarcascade_frontalface_default.xml'))
        d['cascades'].append(cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml'))
        d['cascades'].append(cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml'))
        d['cascades'].append(cv2.CascadeClassifier('./haarcascade_frontalface_alt_tree.xml'))
        d['cascades'].append(cv2.CascadeClassifier('./haarcascade_profileface.xml'))
        d['landmarkDetector'] = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        
        self.__dict__.update(d) 
        
        
    def save( self, filename):
        # save the classifier
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)    
            
    @staticmethod        
    def load( filename) :
        with open(filename, 'rb') as fid:
            return pickle.load(fid)        


    def detect_face(self,image):
        """
        Detect Faces
    
        This function gets image and return the bounding box of the face in the image
        if no face is found the function return None. 
        
        the function use 2 different libraries (dlib, opencv) for the detection 
        with 3 different preprocessings : nothing, histogram equalization & 
        adaptive local histogram equalization (CLAHE)
    
        Parameters
        ----------
        image : np array
            the input image
        
        Returns
        -------
        np.array 
            bounding box of the image in array formatted : [left,right,width,height]
        """   
        for cascade in self.cascades:
            if type(cascade) is dlib.dlib.fhog_object_detector:
                bbs = cascade(image)
                if len(bbs) ==0 :
                    bbs = cascade(self.clahe.apply(image))
                    if len(bbs) > 0 :
                        return [bbs[0].left(), bbs[0].top(), bbs[0].right()-bbs[0].left(),bbs[0].bottom()-bbs[0].top()]
                    else :
                        bbs = cascade(cv2.equalizeHist(image))
                        if len(bbs) > 0 :
                            return [bbs[0].left(), bbs[0].top(), bbs[0].right()-bbs[0].left(),bbs[0].bottom()-bbs[0].top()]
                else :
                    return [bbs[0].left(), bbs[0].top(), bbs[0].right()-bbs[0].left(),bbs[0].bottom()-bbs[0].top()]
            else :
                bbs =  cascade.detectMultiScale(
                        image,
                        scaleFactor = 1.1,
                         minNeighbors = 5,
                        minSize = (120, 120),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                if len(bbs) > 0:
                    return list(bbs[0].astype('int'))
                else :
                    bbs =  cascade.detectMultiScale(
                        self.clahe.apply(image),
                        scaleFactor = 1.1,
                         minNeighbors = 5,
                        minSize = (120, 120),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                    if len(bbs) > 0 :
                        return list(bbs[0].astype('int'))
                    else :
                        bbs =  cascade.detectMultiScale(
                        cv2.equalizeHist(image),
                        scaleFactor = 1.1,
                         minNeighbors = 5,
                        minSize = (120, 120),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                        if len(bbs) > 0 :
                            return list(bbs[0].astype('int'))
        return None
        
    def detect_landmark(self,img, bb=None ):
        """
        Detect Face landmarks from image
    
        This function gets image and face bounding box location in the image and
        return 68 face's landmark locations.
        the function uses dlib for ladmarks detection
    
        Parameters
        ----------
        img : np array
            the input image
        bb : np array
            face bounding box, if not geting any use the entire image ad bounding box
        
        Returns
        -------
        np.array 
            face landmarks
        """   
        if bb == None : 
            rect = dlib.rectangle(0,0,img.shape[1],img.shape[0])
        else :
            rect = dlib.rectangle(bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3])
        
        points = self.landmarkDetector(self.clahe.apply(img), rect)
        
        lmarks = []
        for i in range(68):
            lmarks.append((points.part(i).x, points.part(i).y,))
            
        lmarks = np.asarray(lmarks, dtype='float32')
    
        return lmarks    
        
    def noramlize_alignment_rotate(self, img, bb=None):
        """
        align the face to canonical form using rotation and scale transform
    
        This function gets image and face landmarks and use only the eyes to compute
        and apply rotation transform to put any face eyes on streight line
        
        The code concept (with my adaptations) taken from : 
            http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms
        Parameters
        ----------
        img : np array
            the input image
        bb : np array
            face bounding box, if not geting any use the entire image ad bounding box
        
        Returns
        -------
        np.array 
            aligned image
        """ 
        lmarks = np.array(self.detect_landmark(img,bb))
        
        if self.crop ==(-1,-1):
            crop = (img.shape[0],img.shape[1])
        else :
            crop = self.crop
            
        eye_l = (lmarks[37] + lmarks[38] + lmarks[40] + lmarks[41]) * 0.25
        eye_r = (lmarks[43] + lmarks[44] + lmarks[46] + lmarks[47]) * 0.25

        # calculate offsets in original image
        offset_h = math.floor(0.25*crop[0])
        offset_v = math.floor(0.25*crop[1])
        
        # get the direction
        eye_direction = (eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])
        
        
        #calc rotation angle in degrees
        rotation = math.atan2(float(eye_direction[1]),float(eye_direction[0]))
        rotation = rotation * 180/np.pi
        # distance between them
        dx = eye_r[0] - eye_l[0]
        dy = eye_r[1] - eye_l[1]
        dist = math.sqrt(dx*dx+dy*dy)
        
        # calculate the reference eye-width
        reference = crop[0] - 2.0*offset_h
        # scale factor
        scale = float(dist)/float(reference)
        
        rot = cv2.getRotationMatrix2D((eye_l[1],eye_l[0]), rotation,1)
        
        
        res = cv2.warpAffine(img,rot,(img.shape[1],img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
        
        # crop the rotated image
        crop_xy = (int(eye_l[1] - scale*offset_h), int(eye_l[0] - scale*offset_v))
        crop_size = (int(crop[0]*scale), int(crop[1]*scale))
        
        cropped = res[crop_xy[0]:crop_xy[0]+ crop_size[0],crop_xy[1]:crop_xy[1]+ crop_size[1]]

        return cropped

    def noramlize_alignment_affine(self, img, bb=None):
        """
        align the face to canonical form using affine transform
    
        This function gets image and face landmarks and use them to compute
        and apply affine transform to put any face 'parts' in the same place for all
        input images
        s
        The code concept taken from : 
            https://github.com/cmusatyalab/openface/blob/master/openface/align_dlib.py
        Parameters
        ----------
        img : np array
            the input image
        bb : np array
            face bounding box, if not geting any use the entire image ad bounding box
        
        Returns
        -------
        np.array 
            aligned image
        """  
        
        landmarks = self.detect_landmark(img, bb)
        
        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(INNER_EYES_AND_BOTTOM_LIP) 
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],
                                       np.float32(img.shape) * MINMAX_TEMPLATE[npLandmarkIndices])
    
        aligned = cv2.warpAffine(img, H, (img.shape[1],img.shape[0]), borderMode=cv2.BORDER_CONSTANT)
        return aligned        
        
    def align_crop_faces(self, imgs):
        """
        Crop list of images to configured resolution
        
        This function gets list of images, it detect the face on each image,
        pefrorm alignment for each image and then crop the face location and resize
        to specific resolution
        
        Parameters
        ----------
        imgs : list of np.array
            the input images
    
        Returns
        -------
        list of np.array 
            list of cropped-aligned images
        """  
        
        if self.crop ==(-1,-1):
            crop = (img.shape[0],img.shape[1])
        else :
            crop = self.crop
            
        croppedImgs =[]
        print 'detect, align and crop faces'
        bar = progressbar.ProgressBar()
        for imgIdx in bar(range(len(imgs))):
            processedImg = imgs[imgIdx]
        
            newbb = self.detect_face(processedImg)
        
            if newbb != None:
                bb = newbb
            elif bb == None :
                # just in case nothing found and cant take previous as a guess
                bb = [round(processedImg.shape[1]*0.20), round(processedImg.shape[0]*0.10),\
                      round(processedImg.shape[1]*0.80),round(processedImg.shape[0]*0.90)]
                      
            
            
            if (self.pn_method=='affine'):
                processedImg = self.noramlize_alignment_affine(processedImg, bb)   
                smaller = min(processedImg.shape[0],processedImg.shape[1])
                cropped = processedImg[:smaller,:smaller]
                croppedResized= cv2.resize(cropped,(crop[1],crop[0]))
                croppedImgs.append(croppedResized)
            elif (self.pn_method=='rotate'):
                cropped = self.noramlize_alignment_rotate(processedImg, bb)   
                croppedResized= cv2.resize(cropped,(crop[1],crop[0]))
                croppedImgs.append(croppedResized)
            else:
                # take some extras around the detected face:
                bbUse = bb
                difX = int( float(bb[2]*0.1))
                difY = int( float(bb[3]*0.15))
                bbUse[2] = int( float(bb[2]*1.1))
                bbUse[3] = int( float(bb[3]*1.15))
                bbUse[0] = max(bb[0] - math.floor(difX/2),0)
                if (bbUse[0] + bbUse[2]) > processedImg.shape[1] :
                    dif = processedImg.shape[1] - (bbUse[0] + bbUse[2])       
                    bbUse[0] += dif

                bbUse[1] = max(bb[1] - math.floor(difY/2),0)
                if (bbUse[1] + bbUse[3]) > processedImg.shape[0] :
                    dif = processedImg.shape[0] - (bbUse[1] + bbUse[3])
                    bbUse[1] += dif
                cropped = processedImg[bbUse[1]:bbUse[1]+bbUse[3], bbUse[0]:bbUse[0]+bbUse[2]]
                croppedResized= cv2.resize(cropped,(crop[1],crop[0]))
                croppedImgs.append(croppedResized)
                
        return croppedImgs

        
    def spatial_hist( self, img,grid_size_x=8,grid_size_y=8, bins=256, maxVal=0.2):
        """
        Perform spatial histogram on image
        
        This function gets image and params and calculate histogram of each region
        in the grid (non overlapping patches), The number of grids decided by grid_size_x
        and grid_size_y params.
        All the histogram calculated are concatinate to one vector (for each image)
        this vector represent the "image-signature"
        
        Parameters
        ----------
        img : np.array
            the input image
        grid_size_x: int
            grid resolution in X axis
            default: 8
        grid_size_y: int 
            grid resolution in Y axis
            default: 8
        bins : int 
            Number of bins for the histogram
            default: 256
        maxVal: float
            Maximum allowed value for values in the histogram, used for better normaliation
            default: 0.2
        Returns
        -------
        np.array 
            the spatial histogram : concatination of all grid histograms (AKA face-signature)
        """  
        h = int(np.ceil(img.shape[0] / grid_size_y))
        w = int(np.ceil(img.shape[1] / grid_size_x))
        hists =[]

        for i in range(0,img.shape[0], h):
            for j in range(0,img.shape[1], w):
                region = img[i:min(i+h,img.shape[0]),j:min(j+w,img.shape[1])]
                if (self.hist_norm == 'hassner'):
                    # this is the wierd type of norm I found in  their implementation
                    #take hist:
                    hist = np.histogram(region.flatten(), bins=bins, range=(0, bins), normed=False)[0] 
                    # and normalize :
                    hist = hist / np.sqrt( (hist**2).sum() )
                    hist[hist>0.2] = 0.2
                    hist = hist / np.sqrt( (hist**2).sum() )
                elif (self.hist_norm=='hellinger') :
                    hist = np.histogram(region.flatten(), bins=bins, range=(0, bins), normed=False)[0] 
                    # apply the Hellinger kernel by first L1-normalizing and taking the
                    # square-root
                    hist = hist / (hist.sum()+ self.eps)
                    hist = np.sqrt(hist)
                else:
                    # or just standard norm (unit vec)
                    hist = np.histogram(region.flatten(), bins=bins, range=(0, bins), normed=True)[0] 
                    
                hists.extend(hist)
        
        return np.asarray(hists).ravel()
        
    def homomorphic_filtering(self, img, high=1.1, low=0.5, D0=15,c=1.2):
        """
        Perform homomorphic filtering to the image
        
        Implemented as described in : "Face recognition under varying illumination based on
        adaptive homomorphic eight local directional patterns" by Faraji et al.
        http://digital.cs.usu.edu/~xqi/Promotion/IETCV.FR.14.pdf
        
        Parameters
        ----------
        img : np.array
            the input image
        high: float
            The alpha for high frequencies
        low: float
            The alpha for low frequencies
        D0: float
            the cut-off radius for highpass filter
        c: float
            constant that control the sharpness of the slope
        Returns
        -------
        np.array 
            image after homomorphic filtering (hopefully illumination normalized)
        """  
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
  
        # Set scaling factors and add
        Iout = np.real(fp.ifft2(fftImg*Hshift))
    
        # Anti-log then rescale to [0,1]
        Ihmf = np.expm1(Iout)
        Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
        Ihmf2 = np.array(255*Ihmf, dtype="uint8")
        return Ihmf2
        
    def TanTriggs( self, img, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1.0, sigma1 = 2.0):
        """
        Perform Tan&Triggs illumination normalization algorthm on input image
        
        based on: https://lear.inrialpes.fr/pubs/2007/TT07/Tan-amfg07a.pdf
        
        Parameters
        ----------
        img : np.array
            the input image
        alpha : float
			a strongly compressive exponent that reduces the influence of large values
        tau: float
			threshold used to truncate large values after the first phase of normalization
        gamma : float
			gamma for the gamma correction
        sigma0: float
			lower std for the DoG filtering
        sigma1: float
			higher std for the DoG filtering
        Returns
        -------
        list of np.array 
            list of cropped-aligned images
        """  
    
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
    
    def chi2_d(self, a, b):
        """
        Chi-square distance
        
        This function gets two vectors and calculate their chi-square distance
        
        Parameters
        ----------
        a : np.array
            first vector
        b: np.array
            second vector
        Returns
        -------
        float
            chi-square distance between a and b
        """  
        eps = 1e-10    
           # compute the chi-squared distance
        return 0.5 * np.sum([((a- b) ** 2) / (a+ b+ eps)])   
        
    def makePatchSampleCoordMatrix(self, YY,XX,rows,cols,patchRadius):
        """
        Helper function for TPLBP
        
        """  
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
        
    def tplbp(self,img,S=8, radius=2, w=3, alpha=5, tau=0.01):
        """
        calculate Three-Patch LBP features
        
        This function gets image and some params and calculate Three-patch lbp features
        from it based on "Descriptor Based Methods in the Wild" by Wolf et al:
            
        see more explains in the report
        
        Parameters
        ----------
        img : np.array
            input image
        w: int
            patch size
        radius: int
            neighbors radius
        S : int
            Number of neighbors
        alpha: int
            patch dif size
        Returns
        -------
        float
            chi-square distance between a and b
        """  
        imgf = np.float64(img)
        
        patchRadius = int(math.floor(w/2))
        rows = img.shape[0]
        cols = img.shape[1]
        border = radius + 2*patchRadius
        [XXbase,YYbase]=np.meshgrid(range( cols),range(rows));
        
        XX = XXbase[border:-border,border:-border].ravel('F')
        YY = YYbase[border:-border,border:-border].ravel('F')
        ii= XX*rows + YY
        
        indSample = self.makePatchSampleCoordMatrix(YY,XX,rows,cols,patchRadius)
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
            indSample = self.makePatchSampleCoordMatrix(YYcircles[:,ni],XXcircles[:,ni], rows, cols, patchRadius)
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
        
    def ltp(self, img, t=0.1):
        imgf = np.float64(img)
        
        low_ltp = np.zeros(imgf.shape,'uint8')
        high_ltp = np.zeros(imgf.shape,'uint8')
        for i in range(1,imgf.shape[0]-1):
            for j in range(1,imgf.shape[1]-1):
                center = imgf[i,j]
                region = imgf[i-1:i+2,j-1:j+2]
                
                out = np.zeros((3,3))
                low = center - t
                high = center +t
                out[region< low] = -1
                out[region > high] = 1

                upper = out.copy()
                upper[upper==-1] = 0

                lower = out.copy()
                lower[lower==1] = 0
                lower[lower==-1] = 1

                high_ltp[i,j]= int(''.join(list(upper.astype('int').astype('str').flatten())),2)
                low_ltp[i,j]= int(''.join(list(lower.astype('int').astype('str').flatten())),2)
        return (low_ltp,high_ltp)
                

    def normalize_illumination(self, imgs):
        """
        Run illumination normalization algorithm on list of images
        
        two methods are supported 
            'tantriggs' - perform algorithm developed by Tan&Triggs
                          https://lear.inrialpes.fr/pubs/2007/TT07/Tan-amfg07a.pdf
            'homomorphic' - perform homomorphic filtering to the image
            
        Parameters
        ----------
        imgs : list of np.array
            list of input images
        
        Returns
        -------
        list of np.array
            images after illumination normalization
        """          
        if self.in_method == 'tantriggs':
            in_norm = self.TanTriggs
        elif self.in_method == 'homomorphic':
            in_norm = self.homomofphic_filtering
        elif self.in_method=='equalizehist':
            in_norm = cv2.equalizeHist
        else :
            return imgs
            
        normed_imgs = []
        for img in imgs:
            normed_imgs.append(in_norm(img))
            
        return normed_imgs
    
    def extract_features(self,imgs):
        """
        Extract features from list of images
        support 5 methods : 'lbpror_h', 'lbp_h', 'tplbp_h', 'tplbp_lbpror_h', 'ltp_h'
            
        Parameters
        ----------
        imgs : list of np.array
            list of input images
        y : list
            labels of the images
        """ 
    
        features = []
         
        if self.features == 'lbpror_h' :
            for img in imgs:
                features.append(self.spatial_hist(
                            feature.local_binary_pattern(img,
                            self.S,self.radius, method='ror'), self.grid_x, self.grid_y))
             
        elif self.features == 'tplbp_h' :
            for img in imgs:
                features.append(self.spatial_hist(
                            self.tplbp(img,
                            self.S,self.radius), self.grid_x, self.grid_y))
            
        elif self.features == 'tplbp_lbpror_h' :
            for img in imgs:
                hist1 = self.spatial_hist(feature.local_binary_pattern(img,
                            self.S,self.radius, method='ror'), self.grid_x, self.grid_y)
                    
                hist2 = self.spatial_hist(self.tplbp(img,
                            self.S,self.radius), self.grid_x, self.grid_y)
            
                # combine histograms for two methods
                hist = np.hstack((hist1, hist2))
                features.append(hist)
                
        elif  self.features == 'ltp_h' :
            for img in imgs:
                (low,high) = self.ltp(img)
                hist1 = self.spatial_hist(low, self.grid_x, self.grid_y)
                hist2 = self.spatial_hist(high, self.grid_x, self.grid_y)
                # combine histograms for two methods
                hist = np.hstack((hist1, hist2))
                features.append(hist)
        else:
            for img in imgs:
                features.append(self.spatial_hist(
                            feature.local_binary_pattern(img,
                            self.S,self.radius), self.grid_x, self.grid_y))
                
            
        return np.vstack(features)
    
    def process_imgs(self, imgs):
        """
        combine all the process from input images to feature for the classifier
        
        Parameters
        ----------
        imgs : list of np.array
            list of input images
        
        Returns
        -------
        array of the images as features(face signatures)
            
        """    
        # Start with detect align and crop(if nessecary):
        res = imgs[0].shape

        #ybinarized = label_binarize(y)
        
        if (self.crop != (-1,-1) and self.crop != res ):
            cropped_imgs = self.align_crop_faces(imgs)
        else :
            cropped_imgs = imgs
        
        # illumination normalization:
        if (self.in_method != 'none'):
            normalized_imgs= self.normalize_illumination(cropped_imgs)
        else :
            normalized_imgs = cropped_imgs
            
        return self.extract_features(normalized_imgs)
        
    def train(self, imgs, y):
        """
        Train the model from input images with labels
            
        Parameters
        ----------
        imgs : list of np.array
            list of input images
        y : list
            labels of the images
        """ 
        
        # detect-align-crop -> illumination normalization->extract features-> learn model
        features = self.process_imgs(imgs)
        
        self.dictionary_ = features
        self.ydictionary_ = np.array(y)
        # learn model
        self.classifier.fit(features, y)
        
        # finetune to specific pairs(25->22),(7->5):
        if self.finetune==1:
            y_np = np.array(y)
            x7 = features[y_np==7]
            x5 = features[y_np==5]

            x7_5=np.vstack((x7,x5))     
            y7_5 = [0]*len(x7) + [1]*len(x5)
            
            self.classifier7_5.fit(x7_5, y7_5)            
            x25 = features[y_np==25]
            x22 = features[y_np==22]

            x25_22=np.vstack((x25,x22))     
            y25_22 = [0]*len(x25) + [1]*len(x22)
            self.classifier25_22.fit(x25_22, y25_22)            
            
            
    
    def predict(self, imgs):
        """
        Predict identities for set of images
        
        Parameters
        ----------
        imgs : list of np.array
            list of input images
        
        Returns
        -------
        array of prediction labels for the input images
            
        """    
        features = self.process_imgs(imgs)
        
        y= self.classifier.predict(features)
        
        if self.finetune==1:
            y_np = np.array(y)
            
            idx7 = np.where(y_np==7)[0]
            idx25 = np.where(y_np==25)[0]
            
            y7_5 = self.classifier7_5.predict(features[idx7])
            y25_22 = self.classifier25_22.predict(features[idx25])
            
            idx7to5 = idx7[y7_5==1]
            idx25to22 = idx25[y25_22==1]
            y_np[idx7to5] = 5
            y_np[idx25to22] = 22

            y = y_np
            
        
        return y
        
    
    def predict_proba(self,imgs):
        """
        Predict identities for set of images and return probabilities
        
        Parameters
        ----------
        imgs : list of np.array
            list of input images
        
        Returns
        -------
        array of probabilities for the input images
            
        """    
        features = self.process_imgs(imgs)

        
        if type(self.classifier) ==KNeighborsClassifier: 
            #in case of 1-NN classifier we don't have classic threshold ore 
            # score so we create one based on the distance from the nearest neighbor
            # and use this as a probability
            
            n_samples = len(features)
            n_classes = len(self.classifier.classes_)
            
            yscore = np.zeros((n_samples, n_classes))
            
            # take distances from closest neighbor
            dist, neighborIdx= self.classifier.kneighbors(features)
            
            label = self.ydictionary_[neighborIdx]
            #convert distances to score (apply 1/dist and normalize)
            score = 1/(dist+self.eps)
            
            #score *= (1/score.max()) # try without normalizing
            
            for i in range(n_samples):
                yscore[i, label[i]] = score[i]
            
            pass
        else :
            yscore= self.classifier.predict_proba(features)
        
        if self.finetune==1:
            # Don't sure how to apply finetune to prediction probability
            pass
        
        return yscore
        