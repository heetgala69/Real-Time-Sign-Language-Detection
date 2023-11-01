# Real-Time-Sign-Language-Detection
The "Real-time Sign Language Detection" project is designed to detect and recognize hand signs in real-time using computer vision and machine learning techniques. The project aims to assist individuals with hearing impairments by providing a tool that can interpret sign language gestures and display the corresponding text or spoken words. This introduction provides an overview of the project's objectives, the tools and technologies used, and its functionality.

# Features
1. Collect images for building a sign language dataset.
2. Train a sign language detection model using Google's Teachable Machine.
3. Real-time sign language gesture detection using your webcam.

# Requirements
- Python 3.x
- OpenCV
- Mediapipe
- cvzone
- Tensorflow

# Usage 
 1. Install the necessary Python libraries installed using pip :
   - pip install opencv-python
   - pip install mediapipe
   - pip install cvzone
   - pip install tensorflow
   
 2.Clone the Repository:
  git clone https://github.com/heetgala69/Real-Time-Sign-Language-Detection 
  - cd Real-Time-Sign-Language-Detection

 3.If you want to collect Images run to increase the mode Data then run :
  - python datacollection.py 

 4.If you want to train the model then go to upload you data :
  https://teachablemachine.withgoogle.com/train

 5.Then replace you model downloaded in test code on line number 9 "classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")" 

 6.Then run : 
  - python test.py

 7. Then show the respective signs photos of signs are also provided

# Acknowledgments
This project was developed by Heet Gala as part of Semester 5 Software Engineering Project under the guidance of Prof. Ruchi Sharma, Mukesh Patel School of Technology Management & Engineering (Mumbai), NMIMS.

# Contact
If you have any questions or suggestions, feel free to contact me at:
* Heet Gala - Heet.Gala018@nmims.edu.in
