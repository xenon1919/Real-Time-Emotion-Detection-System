# Real-Time Emotion Detection System with Facial Expression Recognition

## Overview:
This project aims to develop a real-time emotion detection system that operates on streaming video data. The system identifies the predominant emotion in each frame of the video using facial expression recognition techniques. To achieve this, we utilize machine learning, specifically convolutional neural networks (CNN), implemented in Keras, a high-level neural networks API.

## Key Components and Technologies Used:
- **Machine Learning Frameworks**:
  - TensorFlow: Utilized as the backend for Keras to accelerate neural network computations.
  - Keras: Employed for building and training the convolutional neural network (CNN) model for facial expression recognition.

- **Data Visualization**:
  - Matplotlib: Used for visualizing data, including model performance metrics and sample images.

- **Graphical User Interface (GUI)**:
  - Tkinter: Employed to develop the GUI for real-time video streaming and emotion detection.

- **Image Processing**:
  - OpenCV: Utilized for capturing and processing video frames, as well as for integrating the real-time video feed into the GUI.

- **Convolutional Neural Network (CNN)**:
  - Implemented a CNN architecture to learn and recognize facial expressions from image data.

- **Feeding Dataset into Machine Learning Model**:
  - Utilized datasets sourced from online platforms like Kaggle to train and validate the facial expression recognition model.

- **Development Environment**:
  - Anaconda: Used as the primary Python distribution to manage packages and environments.
  - Jupyter Notebook: Employed for interactive development, experimentation, and documentation.

## Project Workflow:
1. **Data Acquisition and Preprocessing**:
   - Obtained facial expression datasets from platforms like Kaggle.
   - Preprocessed the data to prepare it for training.

2. **Model Development**:
   - Designed and implemented a CNN architecture using Keras for facial expression recognition.
   - Trained the model using the preprocessed dataset.

3. **Real-Time Emotion Detection**:
   - Integrated the trained model into a real-time video streaming pipeline.
   - Utilized OpenCV for capturing and processing frames from the video feed.
   - Applied the trained model to detect and classify emotions in each frame.

4. **GUI Development**:
   - Developed a user-friendly GUI using Tkinter to display the real-time video feed and detect emotions.

5. **Testing and Deployment**:
   - Tested the system for accuracy and real-time performance.
   - Deployed the system, allowing users to interact with the GUI and experience real-time emotion detection.


## Contributors:
- Rishi Sai Teja Ramanchi
