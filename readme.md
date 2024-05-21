# Pizza vs. Steak Classification using Tiny VGG Architecture
## Introduction
This project aims to classify images of pizza and steak using a convolutional neural network (CNN) based on the Tiny VGG architecture. The classification task involves determining whether an input image contains pizza or steak.

## Dataset
The dataset used for training and testing consists of images of pizza and steak. It is divided into two sets: a training set and a test set. Each set contains an equal number of images for pizza and steak.

## Training Set: 
This set is used to train the neural network. It includes a balanced number of pizza and steak images to ensure that the model learns to classify both classes effectively.

## Test Set: 
This set is used to evaluate the performance of the trained model. It also contains an equal number of pizza and steak images.

## Model Architecture
The neural network architecture employed for classification is based on the Tiny VGG architecture, a simplified version of the VGG (Visual Geometry Group) network. The Tiny VGG architecture consists of a sequence of convolutional layers followed by max pooling layers and fully connected layers.

## Training
The model is trained using the training set. During training, the model learns to extract relevant features from the input images and classify them as either pizza or steak. The training process involves optimizing the model's parameters (weights and biases) to minimize a predefined loss function (binary crossentropy in this case). The Adam optimizer is used to update the model's parameters based on the gradients computed during training.

## Evaluation
After training, the model's performance is evaluated using the test set. The trained model predicts the class (pizza or steak) of each image in the test set, and the predictions are compared against the ground truth labels to measure the model's accuracy.

## Inference
Once trained and evaluated, the model can be deployed to classify new images of pizza and steak. Users can input images into the deployed model, and it will output the predicted class for each image.

# Usage
Install the required dependencies:
```
pip install -r requirements.txt
```
To run the UI
```
streamlit run app.py
```
## Conclusion
The classification of pizza vs. steak images using the Tiny VGG architecture demonstrates the capability of deep learning models to perform image classification tasks. By leveraging convolutional neural networks and appropriate training techniques, accurate classification results can be achieved even for complex image datasets like food classification.