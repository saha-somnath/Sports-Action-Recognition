# Sports-Action-Recognition
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


====
This project has been tested with Histogram of Oriented Gradient (HOG) / Scale Invarient Feature Transfrom (SIFT) methods
and Support Vector Machine (SVM)/ Random Forest classifier.

- Code: SportsActionRecognition_HOG_SVM.py : Implementation of Sports Action Recognition using HOG and SVM classifier.
- Code: SportsActionRecognition_HOG_RF.py : Implementation of Sports Action Recognition using HOG and Random Forest classifier. 
- Code: SportsActionRecognition_SIFT_SVM.py : Implementation of Sports Action Recognition using SIFT and SVM classifier.

 
