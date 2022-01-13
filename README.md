# Driver Drowsiness Detection 😴
A comparative analysis of Classification models (K-Nearest Neighbors and Convolutional Neural Network) is performed in order to classify the drowsiness of a driver. 
Features like Eye Aspect Ratio, Mouth Aspect Ratio, Pupil Circularity and Mouth Aspect Ratio over Eye Aspect Ratio were taken into consideration. This project lies in Classification domain of Machine Learning. 

A study conducted by AAA Foundation for Traffic Safety estimated that 328000 crashes occured annually due to drowsy driving. Among them, over 58% of the injuries are of pedestrians as the drowsy driver lose control and hit them.

# Dataset 📊
### The dataset used in this project can be accessed [here](https://sites.google.com/view/utarldd/home) 

“Real Life Drowsiness Dataset” is used for training and testing the model. This dataset contains around 30 hours of different videos of different individuals and is created by the research team of University of Texas. Each video is around 10 minutes long. They are labeled into one of three classes. The three classes are alert (labeled as 0), low vigilant (labeled as 5) and drowsy (labeled as 10). Numeric values have been used for the labels and the significance for each label is as follows:
- 0 means there is no symptom of drowsiness.
- 5 means transition from awake state to slightly drowsy state.
- 10 means person is feeling drowsy.

_Note:_ AMong these 3 labels (0,5,10) I have only kept 0 and 10 labels as it is a Binary Classification problem.

# Project Flow 🔗
- First, frames are extracted from video dataset at a rate of one frame per second.
- Facial features are extracted from these frames using mlxtend and DLib library. There are around 68 facial landmarks, however, we are only interested in landmarks for the eyes and mouth. 
- The aspect ratio of mouth and eye, along with mouth over eye ratio, is calculated from eye and mouth features for each frame. 
- These features are then fed into VGG-16 and KNN model.


# Prominent Features 🔑
## a. Eye Aspect Ratio (EAR)
The ratio of length and width of eyes is termed as Eye Aspect Ratio. During the drowsiness phase, eyes get smaller, and the person blinks them often, which reduces EAR. If this feature keeps on decreasing during subsequent frames of video, then our model will classify that person in a drowsy class.
_Conclusion:_ EAR decreases – Drowsiness increases
## b. Mouth Aspect Ratio (MAR)
The ratio of length and width of the mouth is termed as Mouth Aspect Ratio. When a person feels drowsy, they tend to yawn more, which increases MAR from the normal condition.
_Conclusion:_ MAR increases – Drowsiness increases
## c. Pupil Circularity (PUC)
This feature emphasis more on the pupil instead of the entire eye. People who feel drowsy will have their half eyes open which will reduce their Pupil Circularity.
## d. Mouth Aspect Ratio over Eye Aspect Ratio (MOE)
As discussed above EAR and MAR are inversely proportional. MAR comes in numerator and EAR comes in the denominator.
_Conclusion:_ MOE increases (MAR increases and EAR decreases) – Drowsiness increases

# Facial Region Index for Key Features
As discussed above we have 68 facial landmarks. AMong them, we are only concerned about eye and mouth. Below mentioned table shows the Index values of these features.

# Classification ALgorithms
## 1. K-Nearest Neighbors
- The dataset used to build KNN model has 17,280 rows and 10 columns.
- 80% of the data is used for trainging and 20% for testing the model.
- In order to choose best value of K, I have used elbow method to find optimal value.
- Scikit-learn library is used to build the model.
- Confusion matrix and Classification report are used to evaluate model performance.
