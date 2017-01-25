import time
import cv2
import os

from sklearn.cluster import KMeans
import sklearn
import matplotlib.cm as cm
from scipy import interp
import dlib
from scipy import ndimage
import bob.ip.facedetect



from sklearn.multiclass import OneVsOneClassifier

from sklearn.mixture import GaussianMixture

def randExample(imgs):
    idx = np.round(np.random.uniform(0,len(imgs))).astype('int')
    plt.figure()
    plt.imshow(imgs[idx],'gray')
    plt.show()
    return imgs[idx]
    




  
def align_images(imgs):
   
    

    
def showGrid( imgs, rows,cols):
    
    plt.figure()
    
    for i in range(rows):
        for j in range(cols):
            idx = j + i*cols
            print idx
            if idx >= len(imgs):
                break
            
            plt.subplot(rows,cols,idx+1)
            plt.imshow(imgs[idx],'gray')
    plt.show()
    


    
    
def load_cropped( basedir):
    data =[]

    labels = os.listdir(basedir)
    for labelIdx in range(len(labels)):
        labelFiles = glob.glob(basedir  + labels[labelIdx] + '/*.jpg')

        loadedFiles = [cv2.cvtColor(plt.imread(file), cv2.COLOR_RGB2GRAY) for file in labelFiles]
        dataLabel = zip(loadedFiles, [int(labelIdx)]*len(loadedFiles))
        
        data.append(dataLabel)
        
    return np.vstack(data), labels

def largest(bbs) :
    if len(bbs) == 0 :
        return None 
    if len(bbs) ==1 :
        return bbs[0]
    
    return max(bbs, key=lambda rect: rect.width() * rect.height())
    

    
#def spatial_histogram(img , int numPatterns, int grid_x, int grid_y, bool normed) {
#    
#    int width = src.shape[0]/grid_x;
#    int height = src.shape[1]/grid_y;
#    // allocate memory for the spatial histogram
#    
#    result = np.zeros(grid_x * grid_y, numPatterns, CV_32FC1);
#    // return matrix with zeros if no data was given
#    if(src.empty())
#        return result.reshape(1,1);
#    // initial result_row
#    int resultRowIdx = 0;
#    // iterate through grid
#    for(int i = 0; i < grid_y; i++) {
#        for(int j = 0; j < grid_x; j++) {
#            Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
#            Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
#            // copy to the result matrix
#            Mat result_row = result.row(resultRowIdx);
#            cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
#            // increase row count in result matrix
#            resultRowIdx++;
#        }
#    }
#    // return result as reshaped feature vector
#    return result.reshape(1,1);
#}
def calc_dsift(imgs, step_size):
    sift = cv2.xfeatures2d.SIFT_create()
    sifts = []
    for gray in imgs :    
        # extract key points (dense)
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
            for x in range(0, gray.shape[1], step_size)]

        kp2, dense_sift = sift.compute(gray,kp)
#        sifts.append( (dense_sift, sample[1]) )
    
    # stack all train sifts in numpy matrix and return
    return sifts, len(kp)



    

    
def vfv_desc(samples, step_size, codebook, n_centers):
    imgs, labels = zip(*samples)
    
    sifts = calc_dsift(imgs,step_size)
    
    if codebook == None :
        codebook = GaussianMixture(n_components=n_centers, covariance_type='diag', max_iter=20, random_state=0)
        codebook.means_init = np.array([sifts[labels== i].mean(axis=0) for i in range(28)]) #TODO
        codebook.fit(sifts, labels)
        
        
    else :
        pass
        
    
    
def calc_desc(imgs, desc_type, step_size=None, codebook=None):
    if (desc_type=='vfv'):
        return vfv_desc(imgs, step_size, codebook)
    elif desc_type=='edlp':
        res_vec = []
        bar = progressbar.ProgressBar()
        for imgIdx in bar(range(len(imgs))):
            img = imgs[imgIdx]
            res_vec.append(edlp_desc(img).ravel())
        
        return np.vstack(res_vec)
    elif desc_type =='tantriggs':
        TanTriggs
        res_vec = []
        bar = progressbar.ProgressBar()
        for imgIdx in bar(range(len(imgs))):
            img = imgs[imgIdx]
            res_vec.append(TanTriggs(img))
        return res_vec
            
###  SRART OF SCRIPT

# params:
basedir = '/home/dotan/dataset/ExtendedYaleB/'
croppedDir = '/home/dotan/dataset/yaleCropped/'

#train_cropped , labels  = load_cropped(croppedDir + '/train/')
#test_cropped , labels = load_cropped(croppedDir + '/test/')


#train_imgs, ytrain = zip(*train_cropped)
#
#test_imgs, ytest = zip(*test_cropped)
#
#ypredict = np.zeros(np.array(ytest).shape)
#
#train_desc = calc_desc(train_imgs, 'tantriggs')
#
#test_desc = calc_desc(test_imgs, 'tantriggs')
#
#
#
#Xtrain = pca.transform(train_desc)
#Xtest = pca.transform(test_desc)
#
#
#for imgIdx in range(len(Xtest)) :
#    img = Xtest[imgIdx]
#    dist = euclidean_distances(Xtrain,img)
#    resIdx = np.argmin(dist)
#    ypredict[imgIdx] = ytrain[resIdx ]
#    plt.figure()
#    plt.subplot(2,1,1), plt.imshow(test_imgs[imgIdx].reshape(160,125),'gray')
#    plt.subplot(2,1,2), plt.imshow(train_imgs[resIdx].reshape(160,125),'gray')
#    plt.show()
#    
#    
#    

yy = np.zeros((len(test_cropped),1))
for i in range(len(test_cropped)):
    yy[i] = fff.predict(test_cropped[i])

face_res = (100, 100)
dsift_step_size= 2

n_train = 65
#n_train = 2

# load data
train, test, labels = load_dataset(basedir, n_train)


train_imgs, ytrain = zip(*train)

# detect, align and crop faces from train
train_cropped = detect_crop_align(train_imgs, face_res)

# save train cropped
for imgIdx in range(len(train_cropped)):
    label = labels[ytrain[imgIdx]]
    dirname =croppedDir + '/train/' + label
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.imsave(dirname + '/' + str(imgIdx) +'.jpg', train_cropped[imgIdx], cmap=plt.cm.gray)
    
test_imgs, ytest = zip(*test)

test_cropped = detect_crop_align(test_imgs, face_res)

# save test cropped
for imgIdx in range(len(test_cropped)):
    label = labels[ytest[imgIdx]]
    dirname =croppedDir + '/test/' + label
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.imsave(dirname + '/' + str(imgIdx) +'.jpg', test_cropped[imgIdx], cmap=plt.cm.gray)

#
#recog = cv2.face.createLBPHFaceRecognizer()
#recog.train(train_cropped,np.int32(ytrain))


# do preprocessing 
#train_cropped = preprocess(train_cropped,'none')

# calc descriptor
#Xtrain = calc_desc(train_cropped, 'M')

# learn with svm / knn
#clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, algorithm='ball_tree')
#
#clf = NearestNeighbors(n_neighbors=1, n_jobs=-1, algorithm='ball_tree',
#                        metric='pyfunc', func=chi2_distance)
#clf.fit(Xtrain, ytrain)

#model = OneVsRestClassifier(svm.SVC())
#model.fit(Xtrain,ytrain)

# TEST
#test_imgs, ytest = zip(*test)

#test_cropped = detect_crop_align(test_imgs, face_res)

#Xtest = calc_desc(test_cropped, 'edlp')

#print clf.score(Xtest,ytest)




