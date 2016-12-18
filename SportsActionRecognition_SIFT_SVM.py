__author__ = 'somnath'


import numpy as np
import cv2
import sys
import os
import glob
from sklearn import svm



sportsActionPath = "/Users/somnath/MY_PROG/ComputerVision/PA3/ucf_sports_actions/ucf_action"
testPath = "/Users/somnath/MY_PROG/ComputerVision/pa3/Testing/"

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


featuresLimit = 100



'''
Function Name: featureExtraction()
Input Args   : <Sports Action Path>, <Action name>, <Training/ Validation>,
Returns      : <Array: Feature List>
Description  : This function extract features from each frames of a video and consolidated them.
               While it extract features, it add label to feature at the beginning of feature vector based on Sports
               Action Type. It helps to keep tack of feature and corresponding label while shuffle the features during
               cross validation.

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
        # Only take alternate frames
        if  (frameIndex % 2)  == 0 :
            continue

        # Read Frame
        frame = cv2.imread(iFrame)
        # Create SIFT object
        sift = cv2.SIFT()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        fIndex = 0 # Feature Index
        for d in des:
            # Insert Label Only for Training
            if type == "Trng":
                d = np.insert(d, 0, sportsActionTag[actionName])

            videoFeatures.append(d)
            if fIndex >= featuresLimit:
                break
            fIndex += 1


        if frameCount >= 23:
            break
        frameCount += 1



    print " \t\tFrame Count:{0}".format(frameCount)
    #print "Video Features: ", videoFeatures

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
    #print dirs

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
def getSportsActionName(rIndex):

    keys   = sportsActionTag.keys()

    for key in keys:
        if rIndex == sportsActionTag[key]:
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

    # Categories are Rest1=>0, Rest2=> 1, Rest3=>2  etc..
    for fIndex in range(len(truth)):
        '''
        #print "truth-%d predicted-%d" % (int (truth [iDoc]), int( predicted[iDoc][0] ) )
        if ( int(truth[restIndex]) == categoryIndex):
            # TP=> when P[i] = T[i] = Ci
            if (int(truth[restIndex]) == int (predicted[restIndex])):
                TP += 1
            else:
                FP += 1
        elif ( int ( predicted[restIndex]) == categoryIndex ):
            FN += 1
        else:
            TN += 1
        '''
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

    # Calculate Sensitivity - True Positive Rate
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
    subsetLength =  len(featureAndLabelList) / len(sportsActionTag)
    for rIndex in range(len(sportsActionTag)):

        print "INFO: Cross Validation Iteration - ", rIndex
        trainigSet   = []
        valdationSet = []
        feature = []
        label   = []


        if ( rIndex == 0 ):
            trainigSet = featureAndLabelList[1*subsetLength:]
            valdationSet = featureAndLabelList[0: subsetLength]
        elif ( rIndex == 9):
            trainigSet = featureAndLabelList[:9*subsetLength]
            valdationSet = featureAndLabelList[9*subsetLength : ]
        else:
            trainigSet = np.concatenate ((featureAndLabelList[:rIndex * subsetLength] , featureAndLabelList[(rIndex + 1) * subsetLength: ]), axis=0 )
            valdationSet = featureAndLabelList[rIndex * subsetLength : (rIndex + 1 ) * subsetLength]


        # Get all features in a array
        for featureAndLabel in trainigSet:
            label.append(int(featureAndLabel[0]))
            feature.append((np.delete(featureAndLabel, 0)).tolist())

        print "XX:", feature
        print "YY:", label
        #print "Training Feature Length:", len(feature)

        # Train model
        print "INFO: Training "
        clf = svm.SVC(gamma=0.001, C=1.0)
        clf.fit(feature,label)

        # Prepare validation feature and label to be predicted
        print "INFO: Prediction for ", getSportsActionName(rIndex)
        vFeatureList = []
        vLabelList   = [] # Ground Truth
        for featureAndLabel in valdationSet:
            vFeatureList.append(featureAndLabel[1:].tolist())
            vLabelList.append(featureAndLabel[0])

        predictedLabel = clf.predict(vFeatureList)

        # predict validation set and calculate accuracy
        print "INFO: Evaluating ... "
        print "\t Truth - ", vLabelList
        print "\t Predicted - ", str(predictedLabel.tolist())

        # Evaluation < Truth>, <Predicted>, <Sports Action Index>
        (sen, spec , accu ) = evaluation(vLabelList , predictedLabel.tolist() , rIndex)

        sensitivity += sen
        specificity += spec
        accuracy    += accu

        print "\t   Sensitivity : ", sen
        print "\t   Specificity : ", spec
        print "\t   Accuracy    : ", accu



    # Average evaluation metrics
    avgSensitivity = sensitivity / len(sportsActionTag)
    avgSpecificity = specificity / len(sportsActionTag)
    avgAccuracy = accuracy / len(sportsActionTag)


    print "\t*** Overall Evaluation ***"
    print "\t Average Sensitivity: ", avgSensitivity
    print "\t Average Specificity: ", avgSpecificity
    print "\t Average Accuracy   : ", avgAccuracy



def main():
    print "INFO: Action Recognition"

    sportsActionList = getListOfDir( sportsActionPath )
    print "INFO: Sports Action - ",sportsActionList

    sportsActionFeatures = []

    sIndex = 0
    for sportsActionName in sportsActionList:
        sportsActionDir = sportsActionPath + "/" + sportsActionName
        # Get list of videos from each sports action
        videoList = getListOfDir(sportsActionDir)

        print "INFO: Video List:", videoList

        videoFeatures = []
        # For all video in each action category
        for video  in videoList:
            # complete path of video containing jpeg images
            videoPath = sportsActionDir + "/" + video
            print "\tVideo Path:", videoPath
            # Extract Feature
            videoFeatures = featureExtraction(videoPath , sportsActionName, 'Trng')
            #print "Video Features: ", videoFeatures
            # Put together all the videos
            if sIndex == 0:
                sportsActionFeatures = videoFeatures
                sIndex += 1
            else:
                sportsActionFeatures = np.concatenate( (sportsActionFeatures, videoFeatures), axis=0)


    # Cross Validation
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

