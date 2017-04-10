__author__ = 'somnath'


import numpy as np
import cv2
import sys
import os
import glob
from sklearn import svm
from scipy.stats import mode

'''
Program: Sports Action Recognition

Description:
    This program perform the sports action recognition task. First it processes the input videos from UCF sports action
    data set.The data set contains 13 different sports action which individually contains multiple videos. A video
    directory contain a video file and corresponding frames. I iterate over the different sports actions and read video
    frames from each video directory to extract features. I take equal number of videos from each categories.
    Further, to optimize the process, I sorted features with highest gradient for HOG. I have used SVM classifier and
    cross validation for classification and evaluation respectively.

    I have used provided image frame for each video as I found issue to process *.avi file in Mac.



 Feature Extraction: I have used Histogram of Oriented Gradient ( HOG ) method to extract features vector.

                     HOG:  It is constructed by dividing the image into cells and for each cell computing the
                     distribution of intensity gradients or edge directions. The concatenating each of these gradient
                     orientation histograms yields the HOG.

                     hogDescriptor = cv2.HOGDescriptor()
                     hist          = hogDescriptor.compute(gray)

                     I use above two functions to create HOG Descriptor and histogram.
                     Further, I sort the histogram values and take max 15000 values from each frame for evaluation.

 Classifier: I have used Support Vector Machine (SVM) classifier. The classifier parameters are set based on best result
     achieved from different runs. Following are the parameters that has been decided based on the multiple executions.

     Parameters:
      gamma=0.01   Lowering the gamma value gives better result, but takes more time. Optimum value has been chosen.
      C=13
      kernel_type = rbf ( default )
      degree = 3 ( default )

 Evaluation: It is based on K-Fold cross validation mechanism.
               First, I shuffle the feature list which contains features as well as label at the very first element of
               the feature vector to obtain better result. The complete set of shuffled features are divided equally
               into k=13 sub parts. k-1 subset is used for training and one subset is used for validation. I iterate the
               process for k=13 times with different subset combinations for training and validation.

               Evaluation Metrics:
                At each iteration, evaluation metrics sensitivity, specificity and accuracy are calculated
                based on True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN) rates.

                Sensitivity = ( True Positive Rate) = TP / ( TP + FN )
                Specificity = ( True Negative Rate) = TN / ( TN + FP )
                Accuracy    = ( TP + TN ) / ( TP + FN + FP + TN )

               At the end of all iterations of cross validation, I average them all to get average rate.


 Testing: I also have tested my model to check if that works with unseen data or videos.
          For that, I have taken one video from "Diving-Side/014" which has been correctly predicted by my model.
          Result is given below.
'''

sportsActionPath = "/Users/somnath/MY_PROG/ComputerVision/PA3/ucf_sports_actions/ucf_action"
#sportsActionPath = "/Users/somnath/MY_PROG/ComputerVision/pa3/Training"


# Sports Action Tag
sportsActionTag = {
    'Diving-Side': 0,
    'Golf-Swing-Back':1,
    'Golf-Swing-Front':2,
    'Golf-Swing-Side':3,
    'Kicking-Front':4,
    'Kicking-Side':5,
    'Lifting':6,
    'Run-Side':7,
    'SkateBoarding-Front':8,
    'Swing-SideAngle':9,
    'Walk-Front':10,
    'Swing-Bench':11,
    'Riding-Horse':12
}



# Distinct Sports Action Number
sportsActionNumber = len(sportsActionTag)

featuresLimit = 15000


'''
Function Name: featureExtraction()
Input Args   : <Sports Action Path>, <Action name>, <Training/ Validation>,
Returns      : <Array: Feature List>
Description  : This function extract features from each frames of a video and consolidated them.
               While it extract features, it add label to feature at the beginning of feature vector based on Sports
               Action Type. It helps to keep tack of feature and corresponding label while shuffle the features during
               cross validation.

               - I have used histogram of oriented gradient (HOG) method to extract the features.
                 Following methods from cv2 have been used.
                 hogDescriptor = cv2.HOGDescriptor()
                  - It takes default parameter values as Window Size= 64 x 128, block size= 16x16,
                    block stride= 8x8, cell size= 8x8, bins= 9
                 hist = hogDescriptor.compute(gray)
                  - Returns the list of histogram

               - Sorted the Histogram and taken top 15000 for evaluation.
               - I take equal number of image frame from all the videos.
'''
def featureExtraction( videoPath, actionName, type):


    # Set frame path, if jpeg directory doesn't exist , take images from video dir
    framePath = videoPath
    if os.path.exists( framePath + "/jpeg") :
        framePath += "/jpeg/"

    # Extract feature
    imageFrames = getImageList(framePath)
    #print "DEBUG: Image Frames - ", imageFrames

    frameCount = 0
    frameIndex = 0

    # Feature List for a video
    videoFeatures  = []

    for iFrame in imageFrames:

        frameIndex += 1

        # Read Frame
        frame = cv2.imread(iFrame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # HOG Descriptor , default value it takes window size= 64x128, block size= 16x16, block stride= 8x8, cell size= 8x8, bins= 9
        hogDescriptor = cv2.HOGDescriptor()

        # Returns histogram
        hist = hogDescriptor.compute(gray)

        #sortedHogDescriptor = hogDescriptor
        sortedHogHist = np.sort(hist, axis=None)

        keyFeatures = sortedHogHist[- featuresLimit : ]

        if type == "Trng":
            keyFeatures = np.insert(keyFeatures, 0, sportsActionTag[actionName])

        videoFeatures.append(keyFeatures)

        # Lowest number of frame available in a video
        if frameCount >= 23:
            break

        frameCount += 1


    return videoFeatures


'''
Function Name: getImageList()
Input Args   : <Image Directory>
Return       : <Array:List of Images>
Description  : This function returns list of images.
'''
def getImageList(imageDirectory):

    # Find different type of images
    rImages = glob.glob(imageDirectory + "/*.jpg")
    rImages +=  glob.glob(imageDirectory + "/*.jpeg")
    rImages +=  glob.glob(imageDirectory + "/*.png")

    return rImages


'''
Function Name: getListOfDir()
Input Args   : < Path >
Return       : <Array: List of Directory >
Description  : This function returns all the directories under the specified paths
'''
def getListOfDir(path):
    # Read each sport action directory
    dirs  = os.listdir(path)

    sportsActionsCount = 0
    filtered_dir  = []
    # Remove . .. and hidden directory
    for dir in dirs:
        if not dir.startswith("."):
            filtered_dir.append(dir)

    return filtered_dir

'''
Function Name: getSportsActionName()
Input Args   : < Sports Action Index>
Return       : <Sports Action Name>
Description  : This function returns the name of Sports Action based on index value

'''
def getSportsActionName(saIndex):

    keys   = sportsActionTag.keys()

    for key in keys:
        if saIndex == sportsActionTag[key]:
            return key

'''
Function Name: evaluation()
Input Args   : < 1D Array: Truth>, <1D Array: Predicted>, < Sports Action Index>
Return       : <Accuracy>,<Sensitivity>,<Specificity>
Description  :  This function calculate evaluation metrics sensitivity, specificity and accuracy
               based on True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN) rate.

               Sensitivity = ( True Positive Rate) = TP / ( TP + FN )
               Specificity = ( True Negative Rate) = TN / ( TN + FP )
               Accuracy    = ( TP + TN ) / ( TP + FN + FP + TN )

'''

def evaluation( truth, predicted, categoryIndex ):

    # TP,FP,FN,TN indicate True Positive, False Positive, False Negative, True Negative respectively
    TP = 1
    FP = 1
    FN = 1
    TN = 1

    # Categories are Sports Action 1=>0, Sports Action 2=> 1, Sports Action 3=>2  etc..
    for fIndex in range(len(truth)):

         # Positive prediction for each feature
        if ( int(predicted[fIndex]) == categoryIndex):
            # TP=> when P[i] = T[i] = Ci
            if (int(truth[fIndex]) == int (predicted[fIndex])):
                TP += 1
            else:
                FP += 1
        else: # Negative Prediction
            if ( int ( truth[fIndex]) == categoryIndex ):
                FN += 1
            else:
                TN += 1


    # Calculate Sensitivity - True Positive Rate - Recall
    sensitivity = TP / float ( TP + FN )

    # Specificity - True Negative Rate
    specificity = TN / float ( TN + FP )

    #Calculate accuracy
    accuracy =  ( TP + TN ) / float ( TP + FP + FN + TN )


    return sensitivity, specificity, accuracy

'''
Function Name: crossValidation()
Input Args   : < Array: Feature and Label List - Fits element of vector indicates action label and rest are for features>
Retrun       : None
Description  : It perform K-Fold cross validation.
               First, I shuffle the feature list which contains features as well as label at the very first element of
               the feature vector to obtain better result. The complete set of shuffled features are divided equally
               into k=13 sub parts. k-1 subset is used for training and one subset is used for validation. I iterate the
               process for k=13 times with different subset combinations for training and validation.

               Evaluation Metrics:
                At each iteration, evaluation metrics sensitivity, specificity and accuracy are calculated
                based on True Positive (TP), False Positive (FP), False Negative (FN) and True Negative (TN) rates.

                Sensitivity = ( True Positive Rate) = TP / ( TP + FN )
                Specificity = ( True Negative Rate) = TN / ( TN + FP )
                Accuracy    = ( TP + TN ) / ( TP + FN + FP + TN )

               At the end of all iterations of cross validation, I average them all to get average rate.
'''
def crossValidation( featureAndLabelList):

    # Randomize the sample
    np.random.shuffle(featureAndLabelList)


    # Evaluation Metrics
    sensitivity = 0.0
    specificity = 0.0
    accuracy    = 0.0


    # split feature set in equal subsets same as number of sports actions for cross validation
    subsetLength =  len(featureAndLabelList) / sportsActionNumber
    for rIndex in range(sportsActionNumber):

        print "INFO: Cross Validation Iteration - ", rIndex
        trainigSet   = []
        valdationSet = []
        feature = []
        label   = []


        if ( rIndex == 0 ):
            trainigSet = featureAndLabelList[1*subsetLength:]
            valdationSet = featureAndLabelList[0: subsetLength]
        elif ( rIndex == (sportsActionNumber -1) ):
            trainigSet = featureAndLabelList[:(sportsActionNumber -1)*subsetLength]
            valdationSet = featureAndLabelList[(sportsActionNumber -1)*subsetLength : ]
        else:
            trainigSet = np.concatenate ((featureAndLabelList[:rIndex * subsetLength] , featureAndLabelList[(rIndex + 1) * subsetLength: ]), axis=0 )
            valdationSet = featureAndLabelList[rIndex * subsetLength : (rIndex + 1 ) * subsetLength]

        # Get all features in a array
        for featureAndLabel in trainigSet:
            label.append(int(featureAndLabel[0]))
            feature.append((np.delete(featureAndLabel, 0)).tolist())


        # Train model
        print "INFO: Training ... "
        clf = svm.SVC(gamma=0.01, C=13)
        clf.fit(feature,label)

        # Prepare validation feature and label to be predicted
        print "INFO: Prediction for ", getSportsActionName(rIndex)
        vFeatureList = []
        vLabelList   = [] # Ground Truth
        for featureAndLabel in valdationSet:
            vFeatureList.append(featureAndLabel[1:].tolist())
            vLabelList.append(featureAndLabel[0])

        # Predict the class label for Validation Feature List
        predictedLabel = clf.predict(vFeatureList)

        # predict validation set and calculate accuracy
        print "INFO: Evaluating ... "
        #print "\t Truth - ", vLabelList
        #print "\t Predicted - ", str(predictedLabel.tolist())

        # Evaluation < Truth>, <Predicted>, <Sports Action Index>
        (sen, spec , accu ) = evaluation(vLabelList , predictedLabel.tolist() , rIndex)

        sensitivity += sen
        specificity += spec
        accuracy    += accu

        print "\t   Sensitivity : ", sen
        print "\t   Specificity : ", spec
        print "\t   Accuracy    : ", accu


    # Average evaluation metrics
    avgSensitivity = sensitivity / sportsActionNumber
    avgSpecificity = specificity / sportsActionNumber
    avgAccuracy = accuracy / sportsActionNumber


    print "  *** Overall Evaluation ***"
    print "    Average Sensitivity: ", avgSensitivity
    print "    Average Specificity: ", avgSpecificity
    print "    Average Accuracy   : ", avgAccuracy



def main():
    print "INFO: Action Recognition"

    sportsActionList = getListOfDir( sportsActionPath )
    print "INFO: Sports Action - ",sportsActionList

    sportsActionFeatures = []

    firstActionFlag = 0
    for sportsActionName in sportsActionList:
        sportsActionDir = sportsActionPath + "/" + sportsActionName
        # Get list of videos from each sports action
        videoList = getListOfDir(sportsActionDir)

        print "INFO: Video List:", videoList

        videoCount = 1
        videoFeatures = []
        # For all video in each action category
        for video  in videoList:

            # For good result decided to use same number of videos from Action Sports. And same number of frame from each frame
            if videoCount > 5:
                break

            # complete path of video containing jpeg images
            videoPath = sportsActionDir + "/" + video
            print "\tVideo Path:", videoPath

            # Extract Features
            videoFeatures = featureExtraction(videoPath , sportsActionName, 'Trng')

            # Put together all the videos
            if firstActionFlag == 0:
                sportsActionFeatures = videoFeatures
                firstActionFlag = 1
            else:
                sportsActionFeatures = np.concatenate( (sportsActionFeatures, videoFeatures), axis=0)

            videoCount += 1

    ## K-Fold Cross Validation method
    crossValidation(sportsActionFeatures)

    ## ****  Testing with unseen data **** ##

    np.random.shuffle(sportsActionFeatures)
    label = []
    feature = []
    # Get all features in a array
    for featureAndLabel in sportsActionFeatures:
        label.append(int(featureAndLabel[0]))
        feature.append((np.delete(featureAndLabel, 0)).tolist())



    # Train model
    print "INFO: Training ... "
    clf = svm.SVC(gamma=0.01, C=13)
    clf.fit(feature,label)

    # Test Path
    tPath = "/Users/somnath/MY_PROG/ComputerVision/PA3/ucf_sports_actions/ucf_action/Diving-Side/014"
    vFeatures = featureExtraction(tPath , sportsActionName, 'Test')
    predictedLabels = clf.predict(vFeatures)

    #print "Predicted Labels:", predictedLabels
    predictedLabelMode = (mode(predictedLabels))[0]
    print "\t Predicted Sports Action:{0} - {1}".format(predictedLabelMode,getSportsActionName(predictedLabelMode) )


if __name__ == "__main__":
    main()




'''
RESULT:
INFO: Cross Validation Iteration -  0
INFO: Training ...
INFO: Prediction for  Diving-Side
INFO: Evaluating ...
	   Sensitivity :  0.692307692308
	   Specificity :  0.963636363636
	   Accuracy    :  0.934959349593
INFO: Cross Validation Iteration -  1
INFO: Training ...
INFO: Prediction for  Golf-Swing-Back
INFO: Evaluating ...
	   Sensitivity :  0.272727272727
	   Specificity :  0.910714285714
	   Accuracy    :  0.853658536585
INFO: Cross Validation Iteration -  2
INFO: Training ...
INFO: Prediction for  Golf-Swing-Front
INFO: Evaluating ...
	   Sensitivity :  0.5
	   Specificity :  0.965811965812
	   Accuracy    :  0.943089430894
INFO: Cross Validation Iteration -  3
INFO: Training ...
INFO: Prediction for  Golf-Swing-Side
INFO: Evaluating ...
	   Sensitivity :  0.9
	   Specificity :  0.946902654867
	   Accuracy    :  0.943089430894
INFO: Cross Validation Iteration -  4
INFO: Training ...
INFO: Prediction for  Kicking-Front
INFO: Evaluating ...
	   Sensitivity :  0.2
	   Specificity :  0.982300884956
	   Accuracy    :  0.918699186992
INFO: Cross Validation Iteration -  5
INFO: Training ...
INFO: Prediction for  Kicking-Side
INFO: Evaluating ...
	   Sensitivity :  0.1
	   Specificity :  0.982300884956
	   Accuracy    :  0.910569105691
INFO: Cross Validation Iteration -  6
INFO: Training ...
INFO: Prediction for  Lifting
INFO: Evaluating ...
	   Sensitivity :  0.888888888889
	   Specificity :  0.973684210526
	   Accuracy    :  0.967479674797
INFO: Cross Validation Iteration -  7
INFO: Training ...
INFO: Prediction for  Run-Side
INFO: Evaluating ...
	   Sensitivity :  0.583333333333
	   Specificity :  0.90990990991
	   Accuracy    :  0.878048780488
INFO: Cross Validation Iteration -  8
INFO: Training ...
INFO: Prediction for  SkateBoarding-Front
INFO: Evaluating ...
	   Sensitivity :  0.3
	   Specificity :  0.955752212389
	   Accuracy    :  0.90243902439
INFO: Cross Validation Iteration -  9
INFO: Training ...
INFO: Prediction for  Swing-SideAngle
INFO: Evaluating ...
	   Sensitivity :  0.46511627907
	   Specificity :  0.934090909091
	   Accuracy    :  0.892339544513
INFO: Cross Validation Iteration -  10
INFO: Training ...
INFO: Prediction for  Walk-Front
INFO: Evaluating ...
	   Sensitivity :  0.363636363636
	   Specificity :  0.955357142857
	   Accuracy    :  0.90243902439
INFO: Cross Validation Iteration -  11
INFO: Training ...
INFO: Prediction for  Swing-Bench
INFO: Evaluating ...
	   Sensitivity :  0.8
	   Specificity :  0.940677966102
	   Accuracy    :  0.934959349593
INFO: Cross Validation Iteration -  12
INFO: Training ...
INFO: Prediction for  Riding-Horse
INFO: Evaluating ...
	   Sensitivity :  0.9
	   Specificity :  0.902654867257
	   Accuracy    :  0.90243902439
  *** Overall Evaluation ***
    Average Sensitivity:  0.535846909997
    Average Specificity:  0.947984173698
    Average Accuracy   :  0.914169958709


### Testing with unseen data or video which has not been used for training
Test Video: /Users/somnath/MY_PROG/ComputerVision/PA3/ucf_sports_actions/ucf_action/Diving-Side/014

INFO: Training ...
	 Predicted Sports Action:[0] - Diving-Side

'''
