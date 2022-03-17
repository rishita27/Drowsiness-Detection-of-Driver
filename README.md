# Driver Drowsiness Detection 😴
A comparative analysis of Classification models (K-Nearest Neighbors and Convolutional Neural Network) is performed in order to classify the drowsiness of a driver. 
Features like Eye Aspect Ratio, Mouth Aspect Ratio, Pupil Circularity and Mouth Aspect Ratio over Eye Aspect Ratio were taken into consideration. This project lies in Classification domain of Machine Learning. 
This project is Divided into 3 parts:
1. Data Collection and Feature Extraction
2. Build K-Nearest Neighbor and Convolutional Neural Network Model.
3. Compare both the models based on different Evaluation Metrics like Accuracy Precision, Recall, and F1_Score. 

A study conducted by AAA Foundation for Traffic Safety estimated that 328000 crashes occured annually due to drowsy driving. Among them, over 58% of the injuries are of pedestrians as the drowsy driver lose control and hit them.


# Dataset 📊
### The dataset used in this project can be accessed [here](https://sites.google.com/view/utarldd/home) 

“Real Life Drowsiness Dataset” is used for training and testing the model. This dataset contains around 30 hours of different videos of different individuals and is created by the research team of University of Texas. Each video is around 10 minutes long. They are labeled into one of three classes. The three classes are alert (labeled as 0), low vigilant (labeled as 5) and drowsy (labeled as 10). Numeric values have been used for the labels and the significance for each label is as follows:
- 0 means there is no symptom of drowsiness.
- 5 means transition from awake state to slightly drowsy state.
- 10 means person is feeling drowsy.


# Project Flow 🔗
- First, frames are extracted from video dataset at a rate of one frame per second.
- Facial features are extracted from these frames using mlxtend and DLib library. 
- The aspect ratio of mouth and eye, along with mouth over eye ratio, is calculated from eye and mouth features for each frame. 
- These features are then fed into VGG-16 and KNN model.
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/1.png)

# 1. Feature Extraction 
Each video present in the dataset is 10 minutes long and there are total 60 participants. In total there are 30 hours long different videos of different participants. Further, all the 60 participants were randomly divided into five folds of 12 participants, for the purpose of **K-Fold Cross Validation**.
There are around 68 facial landmarks/ facial points, however, we only extracted features of eyes and mouth region.   

Frames were not extracted for first 3 minutes from each video, because, it is the tendancy of human to get distracted in the beginning phase. So to extract meaningful data first 3 minutes from each video were ignored. From 3rd minute onwards 1 frame per second was extracted with the maximum 240 frames per video.  
Below mentioned are prominent features which were evaluated from each photo/frame.  
**Dlib** library was used to extract facial features.

## a. Prominent Features 🔑
### i. Eye Aspect Ratio (EAR) 👀
The ratio of length and width of eyes is termed as Eye Aspect Ratio. During the drowsiness phase, eyes get smaller, and the person blinks them often, which reduces EAR. If this feature keeps on decreasing during subsequent frames of video, then our model will classify that person in a drowsy class.  
__Conclusion:__ EAR decreases – Drowsiness increases
### ii. Mouth Aspect Ratio (MAR)
The ratio of length and width of the mouth is termed as Mouth Aspect Ratio. When a person feels drowsy, they tend to yawn more, which increases MAR from the normal condition.  
__Conclusion:__ MAR increases – Drowsiness increases
### iii. Pupil Circularity (PUC)👁
This feature emphasis more on the pupil instead of the entire eye. People who feel drowsy will have their half eyes open which will reduce their Pupil Circularity.
### iv. Mouth Aspect Ratio over Eye Aspect Ratio (MOE)
As discussed above EAR and MAR are inversely proportional. MAR comes in numerator and EAR comes in the denominator.  
__Conclusion:__ MOE increases (MAR increases and EAR decreases) – Drowsiness increases

## b. Facial Region Index for Key Features 😵
As discussed above, we have total 68 facial landmarks. Among them, we are only concerned about eye and mouth region. Below mentioned table shows the Index values of these facial regions.
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/11.png)
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/2.png)

### c. DataFrame after extracting above mentioned features
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/21.png)

If someone has naturally small eyes, then EAR (Eye Aspect Ratio) will decrease, which will lead our model to classify as a drowsy state. Although that person is alert, our model will fail in that case. To overcome such scenarios, I have performed Standardization for each column/ feature. 

### d. Standardized Dataframe
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/3.png)

### f. Feature Importance
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/4.png)

# 2. Classification Models 
## a. K-Nearest Neighbour
- The dataset used to build KNN model has 17,280 rows and 10 columns.
- 80% of the data is used for trainging and 20% for testing the model.
- **Elbow method** was used to find best value for K.
- Scikit-learn library is used to build the model.
- Confusion matrix and Classification report are used to evaluate model performance.

__Note:__ Among three labels i.e, (0,5,10), I have only kept 0 and 10 for building KNN as it is a binary classification problem.

### Results
#### i. Elbow method to find best value of K
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/6.png)
From the above figure, we can see that at K=5,6,7,and 8 we get a minimum error rate and high accuracy. Since, K=5,6,7,8 shows approx same results, I choose the minimum k value i.e, 5.

#### ii. Confusion Matrix and Classification Report
![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/5.png)

## b. Convolutional Neural Network 
I have used PyTorch Framework to build CNN. Several modules of PyTorch frameworks have been used. `torchvision.transforms` library is used for image transformation. `DataLoader` is used as an iterable object over the dataset. 
  ### i. Architecture of CNN 
  - The Convolutional Neural Network has 2 convolution layers. The first layer has 3 inputs nodes and 10 output nodes with a kernel size of 3 and stride as 1. 
  - The second convolution layer has 10 input and 20 output nodes with the same kernel and stride size. 
  - A Batch normalization layer is added next to regularize the network while training. 
  - A Dropout layer with a dropout percentage of 20% is added to prevent the overfitting of the model. 
  - They are then fed into a single layer feed back network which is again fed into a batch normalization layer. 
  - Another single layer feed back network is used again. 
  - During each forward the tensor will be fed to the forward function where the Maxpooling and flattering is done. 
  - The final output layer uses the softmax function which will give the probability that the given input belongs to class 0(Alert) or Class 1 (Drowsy).

  **Hyperparameters:**
  - Learning rate - 0.001 
  - Number of epochs - 9
  - Activation Function - ReLU
  - Loss function - Cross Entropy Loss function 
  - Optimizer - Stochastic Gradient Descent.
  
  ### ii. Performance Evaluation and Results ✅
  #### a. Graph showing Training Vs Testing Loss
  ![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/7.png)
  From the above figure, we can see that Validation/ Test loss is decreasing as we are increasing the number of epochs. At epoch value of 5 and 8 we have least validation loss.   
  &nbsp;   
  #### b. Training,Testing Loss and Accuracy 
  ![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/8.png)
  From the above figure, it is evident that error is not much significant at the initial values of epochs, however, I have trained the model for 9 iterations just to prevent the problem of overfitting.  
  &nbsp;    
  #### c. Testing the Model on sample Images
  ![image](https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN/blob/main/outputs/9.png)
  In the above figure, first row has original images along with their labels, whereas second row comprises of test results of CNN. We can see that model has correctly predicted/classified all the images.

# 3. Comparative Analysis of Classification Models
- K-Nearest Neighbor gives an accuracy of 83%.
- Convolutional Neural Network gives an accuracy of 99.84%.
- The error rate in Convolutional Neural Network is almost negligible.
- Additional features like **Mouth Aspect Ratio**, **Pupil Circularity**, and **Mouth Aspect Ratio over Eye Aspect Ratio** helped in getting higher accuracy. Most other studies only consider eye aspect ratio.

# How to run code locally 🛠️
- Before following below mentioned steps please make sure you have [git](https://git-scm.com/download), [Anaconda](https://www.anaconda.com/) installed on your system.
- Clone the complete project with  `git clone https://github.com/ManjinderSingh3/Driver-Drowsiness-Detection-using-KNN-and-CNN.git` or you can just download the code and unzip it.
- Create a Pytorch Environment in Anaconda.
- Download the dataset from [here](https://sites.google.com/view/utarldd/home)
- Perform Feature Extraction by following `Feature_Extraction.ipynb` file.
- Frames extracted from videos are saved in `img` folder.
- KNN_Dataset folder has final `normalized_data.csv` file which is used to build KNN Model.
- Follow `K-Nearest Neighbors.ipynb` file to build KNN model.
- As there were huge number of images of training CNN Model so CNN_Dataset folder is currently empty as Gihub allows maximum file size to be 100 Mb. Also Git-lfs has limit of 5 Gb data per month in free trial. 
- To build CNN Model follw `Convolutional Neural Network.ipynb` file.


# Future Scope 🏆
The future work is a real time monitoring system.
- Scope of integrating this code into a mobile application. 
- As almost all the vehicle have a dashcam now, we can use them to capture the video.
- Mobile application will have acces to videos captured by dashcam. 
- Once videos are received in application than feature extraction will be performed.  
- Extracted features will be passed to the highest accurate model. Based on the results of the model, application will notify the driver
- If it founds that driver is drowsy than Mobile Application will start beeping an Alarm to notify the driver and prevent accident.

# Contact 📞

#### If you have any doubt or want to contribute to this project feel free to email me or drop your message on [LinkedIn](https://www.linkedin.com/in/rishita-kotiyal/)
