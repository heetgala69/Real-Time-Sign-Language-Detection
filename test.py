import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["0", "1", "C", "Fire", "Yes"]

# Variables for accuracy calculation
total_frames = 0
correct_predictions = 0

# Initialize confusion matrix
num_classes = len(labels)
confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if not imgCrop.size == 0:  # Check if imgCrop is not empty
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                predicted_label = labels[index]  # Get the predicted label
                print("Prediction:", predicted_label)  # Print the predicted label to the terminal
                print(prediction, index)
                # Print the confusion matrix
                for row in confusion_matrix:
                    print(row)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite)
        else:
            print("imgCrop is empty or has an invalid size")

        total_frames += 1

        # Ground truth label for the current frame (you should set this according to your data)
        ground_truth_label = "C"

        if labels[index] == ground_truth_label:
            correct_predictions += 1

        # Update the confusion matrix
        true_label_index = labels.index(ground_truth_label)
        confusion_matrix[true_label_index][index] += 1

        accuracy = (correct_predictions / total_frames) * 100

        # Display accuracy on the image
        accuracy_text = f"Accuracy: {accuracy:.2f}%"
        cv2.putText(imgOutput, accuracy_text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        # Display confusion matrix on the image as a table
        table_x = imgOutput.shape[1] - 300  # X-coordinate for the top-right corner of the table
        table_y = 10  # Y-coordinate for the top-right corner of the table
        cell_size = 30  # Size of each cell in the table
        cell_padding = 5  # Padding between cells
        table_border_color = (0, 0, 255)
        table_border_thickness = 2

        for i in range(num_classes):
            for j in range(num_classes):
                cell_value = confusion_matrix[i][j]
                cell_x = table_x + j * (cell_size + cell_padding)
                cell_y = table_y + i * (cell_size + cell_padding)
                cv2.rectangle(imgOutput, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), table_border_color, table_border_thickness)
                cv2.putText(imgOutput, str(cell_value), (cell_x + 10, cell_y + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        # Display prediction probabilities in a straight downward line
        prob_x = table_x
        prob_y = table_y + num_classes * (cell_size + cell_padding) + 20  # Below the confusion matrix
        for i in range(len(labels)):
            prob_value = prediction[i]
            prob_color = (0, 255, 0) if i == index else (0, 0, 255)  # Green for the correct label, red for others
            prob_text = f"{labels[i]}: {prob_value:.2f}"
            cv2.putText(imgOutput, prob_text, (prob_x, prob_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, prob_color, 1)
            prob_y += 20  # Adjust this value for proper spacing

        # Display predicted label in the bottom center of the image
        predicted_label_text = f"Predicted Label: {predicted_label}"
        label_x = (imgOutput.shape[1] // 2) - 100  # X-coordinate for the bottom center
        label_y = imgOutput.shape[0] - 40  # Y-coordinate for the bottom center
        label_color = (0, 255, 0) if predicted_label == ground_truth_label else (0, 0, 255)  # Green for correct, red for incorrect
        cv2.putText(imgOutput, predicted_label_text, (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
